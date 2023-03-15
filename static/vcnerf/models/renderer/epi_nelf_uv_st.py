from collections import OrderedDict
import random
import numpy as np
from numpy.core.numeric import outer
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.modules import distance

from torchvision.models import vgg16

from mmcv.runner import auto_fp16, force_fp32
from vcnerf.core import im2mse, mse2psnr, raw2outputs, SamplePDF
from ..builder import RENDERER, build_embedder, build_field


@RENDERER.register_module()
class EPINeLFUvSt(nn.Module):
    def __init__(self, 
                 uv_embedder, 
                 st_embedder, 
                 epi_embedder, 
                 nelf_field, 
                 epi_field, 
                 pixel_move, 
                 near, 
                 far,
                 epi_sample_num,
                 epi_converge_iter, 
                 epi_converge_range,
                 epi_smooth_weight,
                 render_params,):
        super().__init__()
        self.uv_embedder = build_embedder(uv_embedder)
        self.st_embedder = build_embedder(st_embedder)
        self.epi_embedder = build_embedder(epi_embedder)
        self.nelf_field = build_field(nelf_field)
        self.epi_field = build_field(epi_field)
        self.epi_sample_num = epi_sample_num
        self.epi_converge_iter = epi_converge_iter
        self.epi_converge_range = epi_converge_range
        self.epi_smooth_weight = epi_smooth_weight
        self.pixel_move = pixel_move
        self.near = nn.Parameter(torch.tensor(near), requires_grad=False)
        self.far = nn.Parameter(torch.tensor(far), requires_grad=False)
        self.render_params = render_params
        self.fp16_enabled = False
        self.iter = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.max_moving_dis = nn.Parameter(torch.ones([2]), requires_grad=False)
        self.min_moving_dis = nn.Parameter(torch.ones([2]), requires_grad=False)
        self.center_uv = nn.Parameter(torch.tensor([0.,0.]), requires_grad=False)

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if 'loss' not in loss_name:
                continue
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items())
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self, rays, render_params=None):
        """        
        Args:
            rays (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: 
        """
        if render_params is None:
            render_params = self.render_params
        outputs = self.forward_render(**rays, **render_params)

        return outputs

    def _parse_outputs(self, outputs):
        loss, log_vars = self._parse_losses(outputs)
        log_vars['psnr'] = mse2psnr(outputs['rec_loss']).item()
        log_vars['percent'] = self.percent
        outputs.update(dict(loss=loss, log_vars=log_vars))
        outputs['num_samples'] = 1
        return outputs

    @auto_fp16()
    def forward_render(self,
                       uv, st, aug_uv, aug_st, rays_color, 
                       rays_grad=None, h=400, w=400, max_rays_num=-1, 
                       aug_points=-1, perturb=True, without_epi=False):
        if isinstance(h, torch.Tensor):
            h = h[0]; w = w[0]
        self.h, self.w = int(h), int(w)
        b, n, _ = uv.shape
        self.b, self.n = b, n

        uv = uv.reshape([-1,2])
        st = st.reshape([-1,2])
        uv_embeds = self.uv_embedder(uv)
        st_embeds = self.st_embedder(st)
        embeds = [uv_embeds, st_embeds]

        if self.training or not max_rays_num>0:
            nelf_epi, rgb = self.nelf_field(embeds)
            nelf_epi = nelf_epi*(self.far-self.near)+self.near
        else:
            nelf_epi, rgb, epi_e, epi_rgb = self.batch_ray_forward(
                uv, st, embeds, max_rays_num, without_epi)
            nelf_epi = nelf_epi*(self.far-self.near)+self.near
            im_loss = torch.tensor(0., device=rgb.device)
            if rays_color is not None:
                im_loss = im2mse(epi_rgb, rays_color)
                # im_loss = im2mse(rgb, rays_color)
            return {'nelf_color_map': rgb.reshape([b,n,3]), 
                    'epi_color_map': epi_rgb.reshape([b,n,3]), 
                    # 'color_map': rgb.reshape([b,n,3]), 
                    'color_map': epi_rgb.reshape([b,n,3]), 
                    'nelf_epi_map': nelf_epi.reshape([b,n,1]),
                    'epi_map': epi_e.reshape([b,n,1]),
                    'rec_loss': im_loss}
        outputs = {}
        outputs['nelf_color_map'] = rgb.reshape([b,n,3])
        outputs['nelf_epi_map'] = nelf_epi.reshape([b,n,1])

        im_loss = im2mse(rgb, rays_color)
        outputs['nelf_rec_loss'] = im_loss

        # sample epi
        self.update_sample_near_far(nelf_epi.detach())
        sample_epi_expanded, sample_epi_embeds = self.get_epi_sample(perturb)

        # sample_epi = torch.linspace(self.sample_near, self.sample_far, self.running_sample_num*2, device=uv.device)
        # if perturb:
        #     # Gets intervals
        #     mid_points = 0.5 * (sample_epi[1:] + sample_epi[:-1])
        #     upper = torch.cat([mid_points, sample_epi[-1:]], -1)
        #     lower = torch.cat([sample_epi[:1], mid_points], -1)
        #     # stratified samples in those intervals
        #     t_rand = torch.rand(sample_epi.shape, device=sample_epi.device)
        #     sample_epi = lower + (upper-lower)*t_rand
        # sample_epi_embeds = self.epi_embedder(sample_epi.reshape([-1,1]))

        # epi field color rec
        uv_embeds = uv_embeds.expand([self.running_sample_num*2,n,-1])
        # aligned_st_embeds = self.sample_st_by_epi(uv, st, self.center_uv, sample_epi_expanded)
        # return_max = True
        # epi_weight, epi, epi_color = self.forward_epi_field(
        #     uv_embeds, aligned_st_embeds, sample_epi_embeds, sample_epi_expanded, return_max)
        # outputs['rec_loss'] = im2mse(epi_color, rays_color)

        return_max = False
        align_st = self.sample_st_by_epi(uv, st, self.center_uv, sample_epi_expanded, True)
        align_st.requires_grad = True
        aligned_st_embeds = self.st_embedder(align_st)
        epi_weight, epi, epi_color = self.forward_epi_field(
            uv_embeds, aligned_st_embeds, sample_epi_embeds, sample_epi_expanded, return_max)
        if self.training and self.epi_smooth_weight>0:
            epi_output = torch.ones_like(epi, requires_grad=False)
            depi_dst = torch.autograd.grad(outputs=epi, 
                                           inputs=align_st, 
                                           grad_outputs=epi_output, 
                                           create_graph=True)[0]
            outputs['smooth_loss'] = (depi_dst*(-rays_grad).exp()).mean()*self.epi_smooth_weight
        outputs['rec_loss'] = im2mse(epi_color, rays_color)

        # field and epi field consistency
        # 1. on uv,st
        outputs['f_consist_loss'] = ((epi.detach()-nelf_epi)**2).mean()
        # 2. on aug uv,st
        aug_uv = aug_uv.reshape([-1,2])
        aug_st = aug_st.reshape([-1,2])
        aug_uv_embeds = self.uv_embedder(aug_uv)
        aug_st_embeds = self.st_embedder(aug_st)
        aug_e, aug_color = self.nelf_field([aug_uv_embeds, aug_st_embeds])
        aug_e = aug_e*(self.far-self.near)+self.near
        _, epi_e, epi_color = self.forward_epi_field_uv_st(aug_uv, aug_st, return_max)
        outputs['f_consist_loss'] += ((aug_e-epi_e.detach())**2).mean() + \
                                     ((aug_color-epi_color.detach())**2).mean()
        # outputs['f_consist_loss'] *= 0.5
        # outputs['f_consist_loss'] *= 0.
        # outputs['f_consist_loss'] += ((aug_e-epi_e)**2).mean() + ((aug_c-epi_c)**2).mean()
        return outputs

    def sample_st_by_epi(self, ori_uv, ori_st, new_uv, epi, no_embeds=False):
        with torch.no_grad():
            uv_move = new_uv-ori_uv
            # distance = uv_move/(epi.cos()+1e-5)
            # st_move = distance*epi.sin()
            st_move = uv_move*torch.tan(epi)
            new_st = ori_st+st_move
            if no_embeds:
                return new_st
            new_st_embeds = self.st_embedder(new_st)
        return new_st_embeds

    def forward_epi_field(self, 
                          uv_embeds, 
                          st_embeds, 
                          sample_epi_embeds, 
                          sample_epi_expanded,
                          return_epi_argmax=True):
        dist = sample_epi_expanded[1:] - sample_epi_expanded[:-1]
        dist = torch.cat([dist, dist[-1:]],dim=0)
        alpha, epi_color = self.epi_field(uv_embeds, st_embeds, sample_epi_embeds, dist)

        cp = torch.cumprod(1-alpha, 0)
        cp = torch.roll(cp, 1, 0)
        cp[0] = 1.0
        weight = cp*alpha
        epi_color = (weight*epi_color).sum(0)
        if return_epi_argmax:
            epi = sample_epi_expanded.gather(dim=0,index=weight.argmax(0)[None])[0]
        else:
            epi = (weight*sample_epi_expanded).sum(0)
        # epi = torch.clamp(epi, self.near, self.far)
        epi_color = epi_color + torch.rand_like(epi_color)*(1-weight.sum(0)) # random background
        return weight, epi, epi_color

    def forward_epi_field_uv_st(self, uv, st, return_epi_argmax=True):
        n = st.shape[0]
        sample_epi_expanded, sample_epi_embeds = self.get_epi_sample()            
        aligned_st_embeds = self.sample_st_by_epi(uv, st, self.center_uv, sample_epi_expanded)
        uv_embeds = self.uv_embedder(uv)
        uv_embeds = uv_embeds.expand([self.running_sample_num*2,n,-1])
        return self.forward_epi_field(uv_embeds, 
                                      aligned_st_embeds, 
                                      sample_epi_embeds, 
                                      sample_epi_expanded, 
                                      return_epi_argmax)

    def get_epi_sample(self, perturb=False):
        self.running_sample_num = int(self.epi_sample_num*self.percent/2)
        with torch.no_grad():
            n = self.sample_far.shape[0]
            base = torch.linspace(0, 1, self.running_sample_num, device=self.iter.device)
            avg_sample_epi = base.expand([n, self.running_sample_num])
            rand_base = torch.rand([n, self.running_sample_num], device=self.iter.device)
            pred_sample_epi = rand_base*(self.sample_far-self.sample_near)+self.sample_near
            sample_epi = torch.cat([avg_sample_epi, pred_sample_epi], dim=-1)
            sample_epi = sample_epi.sort(dim=-1).values
        # self.running_sample_num = self.epi_sample_num//2
        # with torch.no_grad():
        #     base = torch.linspace(0, 1, self.running_sample_num*2, device=self.iter.device)
        #     sample_epi = base*(self.sample_far-self.sample_near)+self.sample_near

            sample_epi = sample_epi.T[..., None] # [n_epi_sample, n_ray, 1]
            if perturb:
                # Gets intervals
                mid_points = 0.5 * (sample_epi[1:] + sample_epi[:-1])
                upper = torch.cat([mid_points, sample_epi[-1:]], 0)
                lower = torch.cat([sample_epi[:1], mid_points], 0)
                # stratified samples in those intervals
                t_rand = torch.rand(sample_epi.shape, device=sample_epi.device)
                sample_epi = lower + (upper-lower)*t_rand            
            sample_epi_embeds = self.epi_embedder(sample_epi)
        return sample_epi, sample_epi_embeds

    def update_sample_near_far(self, nelf_epi):
        with torch.no_grad():
            if self.iter > self.epi_converge_iter:
                percent = self.epi_converge_range
            else:
                progress = self.iter/self.epi_converge_iter
                percent = 1*(1-progress).item() + (self.epi_converge_range*progress).item()
            self.percent = percent
            self.sample_near = nelf_epi-percent*(self.far-self.near)/2
            small_mask = self.sample_near<self.near
            near_ones = torch.ones_like(self.sample_near)
            self.sample_near[small_mask] = near_ones[small_mask]*self.near
            self.sample_near = self.sample_near*percent + near_ones*(1-percent)*self.near

            self.sample_far = nelf_epi+percent*(self.far-self.near)/2
            large_mask = self.sample_far>self.far
            far_ones = torch.ones_like(self.sample_far)
            self.sample_far[large_mask] = far_ones[large_mask]*self.far
            self.sample_far = self.sample_far*percent + far_ones*(1-percent)*self.far

    def batch_ray_forward(self, uv, st, embeds, max_rays_num, without_epi):
        out_e, out_c = [], []
        epi_out_e, epi_out_c = [], []
        i = 0
        uv_embeds, st_embeds = embeds
        while i < uv_embeds.shape[0]:
            end = min(uv_embeds.shape[0], i+max_rays_num)
            result = self.nelf_field([uv_embeds[i:end,...], st_embeds[i:end,...]])
            self.update_sample_near_far(result[0])
            out_e.append(result[0])
            out_c.append(result[1])
            if not without_epi:
                epi_result = self.forward_epi_field_uv_st(uv[i:end,...], 
                                                          st[i:end,...],
                                                          False)
                epi_out_e.append(epi_result[1])
                epi_out_c.append(epi_result[2])
            i += max_rays_num
        out_e = torch.cat(out_e, dim=0)
        out_c = torch.cat(out_c, dim=0)
        if without_epi:
            epi_out_e = out_e
            epi_out_c = out_c
        else:
            epi_out_e = torch.cat(epi_out_e, dim=0)
            epi_out_c = torch.cat(epi_out_c, dim=0)
        return out_e, out_c, epi_out_e, epi_out_c

    def train_step(self, data, optimizer, **kwargs):
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        kwargs['render_params'] = {'max_rays_num': 1024}
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs



