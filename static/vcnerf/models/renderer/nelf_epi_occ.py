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
class NeLFEPIOcc(nn.Module):
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
                 epi_reg_weight,
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
        self.epi_reg_weight = epi_reg_weight
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
        outputs.update(dict(loss=loss, log_vars=log_vars))
        outputs['num_samples'] = 1
        return outputs

    @auto_fp16()
    def forward_render(self,
                       uv, st, aug_uv, aug_st, rays_color, all_uv, 
                       rays_grad=None, h=400, w=400, batch_ray_forward=False, 
                       aug_points=-1, perturb=True, without_occ=False):
        if isinstance(h, torch.Tensor):
            h = h[0]; w = w[0]
        self.h, self.w = int(h), int(w)
        all_uv = all_uv.reshape([-1,2])
        b, n, _ = uv.shape
        self.b, self.n = b, n

        st.requires_grad_(True)
        uv_embeds = self.uv_embedder(uv.reshape([-1,2]))
        st_embeds = self.st_embedder(st.reshape([-1,2]))
        embeds = [uv_embeds, st_embeds]

        if self.training or not batch_ray_forward:
            nelf_epi, rgb = self.nelf_field(embeds)
            nelf_epi = nelf_epi*(self.far-self.near)+self.near
        else:
            nelf_epi, rgb, occ_e, occ_rgb = self.batch_ray_forward(embeds, batch_ray_forward, without_occ)
            nelf_epi = nelf_epi*(self.far-self.near)+self.near
            im_loss = torch.tensor(0., device=rgb.device)
            if rays_color is not None:
                im_loss = im2mse(rgb, rays_color)
            return {'color_map': rgb.reshape([b,n,3]), 
                    'occ_color_map': occ_rgb.reshape([b,n,3]), 
                    'epi_map': nelf_epi.reshape([b,n,1]),
                    'occ_epi_map': occ_e.reshape([b,n,1]),
                    'rec_loss': im_loss}
        outputs = {}
        outputs['color_map'] = rgb.reshape([b,n,3])
        outputs['epi_map'] = nelf_epi.reshape([b,n,1])

        im_loss = im2mse(rgb, rays_color)
        outputs['nelf_rec_loss'] = im_loss

        # sample epi
        self.update_sample_near_far(nelf_epi.detach())
        sample_epi_expanded, sample_epi_embeds = self.get_epi_sample(perturb)

        # sample_epi = torch.linspace(self.sample_near, self.sample_far, self.epi_sample_num, device=uv.device)
        # if perturb:
        #     # Gets intervals
        #     mid_points = 0.5 * (sample_epi[1:] + sample_epi[:-1])
        #     upper = torch.cat([mid_points, sample_epi[-1:]], -1)
        #     lower = torch.cat([sample_epi[:1], mid_points], -1)
        #     # stratified samples in those intervals
        #     t_rand = torch.rand(sample_epi.shape, device=sample_epi.device)
        #     sample_epi = lower + (upper-lower)*t_rand
        # sample_epi_embeds = self.epi_embedder(sample_epi.reshape([-1,1]))

        # occ field color rec
        uv_embeds = uv_embeds.expand([self.epi_sample_num,n,-1])
        st_embeds = st_embeds.expand([self.epi_sample_num,n,-1])
        # sample_epi_embeds = sample_epi_embeds.expand([n,self.epi_sample_num,-1]).permute([1,0,2])
        # sample_epi_expanded = sample_epi.expand([n,self.epi_sample_num]).T[...,None]
        epi_weight, epi, epi_color = self.forward_epi_field(
            uv_embeds, st_embeds, sample_epi_embeds, sample_epi_expanded, False)
        if self.training:
            epi_output = torch.ones_like(epi, requires_grad=False)
            depi_dst = torch.autograd.grad(outputs=epi, 
                                           inputs=st, 
                                           grad_outputs=epi_output, 
                                           create_graph=True)[0]
            outputs['smooth_loss'] = (depi_dst*(-rays_grad).exp()).mean()*self.epi_smooth_weight

        # outputs['epi_rec_loss'] = im2mse(epi_color, rays_color)
        outputs['rec_loss'] = im2mse(epi_color, rays_color)
        # outputs['epi_weight_loss'] = ((epi_weight.sum(0)-1)**2).mean()

        # occ field consistency
        # 1. sample other rays; rays on other uv grid
        # 2. consistency loss (l2)
        # sample_uv = all_uv[int(random.random()*all_uv.shape[0])].expand_as(uv)
        sample_uv = torch.stack(random.choices(all_uv, k=uv.shape[1]), dim=0)
        sample_uv, sample_st = self.sample_st_by_epi(uv, st, sample_uv, sample_epi_expanded)
        s_weight, s_epi, s_color = self.forward_epi_field(
            sample_uv, sample_st, sample_epi_embeds, sample_epi_expanded)
        outputs['epi_aug_rec_loss'] = im2mse(s_color, rays_color)

        # 3. sample other rays; rays on random uv (aug uv)
        # 4. consistency loss (l2)
        sample_uv, sample_st = self.sample_st_by_epi(uv, st, aug_uv, sample_epi_expanded)
        s_weight, s_epi, s_color = self.forward_epi_field(
            sample_uv, sample_st, sample_epi_embeds, sample_epi_expanded)        
        outputs['epi_aug_rec_loss'] += im2mse(s_color, rays_color)

        # field and occ field consistency
        # 1. on uv,st
        outputs['f_consist_loss'] = ((epi.detach()-nelf_epi)**2).mean()

        # 2. on aug uv,st
        aug_uv_embeds = self.uv_embedder(aug_uv.reshape([-1,2]))
        aug_st_embeds = self.st_embedder(aug_st.reshape([-1,2]))
        aug_e, aug_color = self.nelf_field([aug_uv_embeds, aug_st_embeds])
        aug_e = aug_e*(self.far-self.near)+self.near
        _, epi_e, epi_color = self.forward_epi_field(aug_uv_embeds, aug_st_embeds)
        outputs['f_consist_loss'] += ((aug_e-epi_e.detach())**2).mean() + \
                                     ((aug_color-epi_color.detach())**2).mean()
        # outputs['f_consist_loss'] += ((aug_e-epi_e)**2).mean() + ((aug_c-epi_c)**2).mean()
        return outputs

    def sample_st_by_epi(self, ori_uv, ori_st, new_uv, epi):
        with torch.no_grad():
            uv_move = new_uv-ori_uv
            # distance = uv_move/(epi.cos()+1e-5)
            # st_move = distance*epi.sin()
            st_move = uv_move*torch.tan(epi)
            new_st = ori_st+st_move
            new_uv = new_uv.expand_as(new_st)
            new_uv_embeds = self.uv_embedder(new_uv)
            new_st_embeds = self.st_embedder(new_st)
        return new_uv_embeds, new_st_embeds

    def forward_epi_field(self, 
                          uv_embeds, 
                          st_embeds, 
                          sample_epi_embeds=None, 
                          sample_epi_expanded=None,
                          return_epi_argmax=True):
        if sample_epi_embeds is None: # forward with uv-st
            n = uv_embeds.shape[0]
            uv_embeds = uv_embeds.expand([self.epi_sample_num,n,-1])
            st_embeds = st_embeds.expand([self.epi_sample_num,n,-1])
            sample_epi_expanded, sample_epi_embeds = self.get_epi_sample()            
            # sample_epi = torch.linspace(self.sample_near, self.sample_far, self.epi_sample_num, device=uv_embeds.device)
            # sample_epi_embeds = self.epi_embedder(sample_epi.reshape([-1,1]))
            # sample_epi_embeds = sample_epi_embeds.expand([n,self.epi_sample_num,-1]).permute([1,0,2])
            # sample_epi_expanded = sample_epi.expand([n,self.epi_sample_num]).T[...,None] # [k,n,1]
        # scale = sample_epi_expanded[1:,:,:]-sample_epi_expanded[:-1,:,:]
        # scale = torch.cat([scale, 1e8*torch.ones_like(scale[:1,...])], dim=0)
        # scale = torch.cat([scale, torch.ones_like(scale[:1,...])], dim=0)
        scale = 1/self.epi_sample_num
        alpha, epi_color = self.epi_field(uv_embeds, st_embeds, sample_epi_embeds, scale)

        cp = torch.cumprod(1-alpha, 0)
        cp = torch.roll(cp, 1, 0)
        cp[0] = 1.0
        weight = cp*alpha
        epi_color = (weight*epi_color).sum(0)
        if return_epi_argmax:
            epi = sample_epi_expanded.gather(dim=0,index=weight.argmax(0)[None])
        else:
            epi = (weight*sample_epi_expanded).sum(0)
        # epi = torch.clamp(epi, self.near, self.far)
        epi_color = epi_color + torch.rand_like(epi_color)*(1-weight.sum(0)) # random background
        return weight, epi, epi_color

    def get_epi_sample(self, perturb=False):
        with torch.no_grad():
            base = torch.linspace(0, 1, self.epi_sample_num, device=self.iter.device)
            sample_epi = base*(self.sample_far-self.sample_near)
            sample_epi += self.sample_near
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
            progress = min(self.iter, self.epi_converge_iter)/self.epi_converge_iter
            percent = 1*(1-progress) + self.epi_converge_range*progress
            self.sample_near = nelf_epi-percent*(self.far-self.near)/2
            small_mask = self.sample_near<self.near
            near_ones = torch.ones_like(self.sample_near)
            self.sample_near[small_mask] = near_ones[small_mask]*self.near
            self.sample_near = self.sample_near*progress + near_ones*(1-progress)*self.near

            self.sample_far = nelf_epi+percent*(self.far-self.near)/2
            large_mask = self.sample_far>self.far
            far_ones = torch.ones_like(self.sample_far)
            self.sample_far[large_mask] = far_ones[large_mask]*self.far
            self.sample_far = self.sample_far*progress + far_ones*(1-progress)*self.far

    def batch_ray_forward(self, embeds, batch_ray_forward, without_occ):
        out_e, out_c = [], []
        epi_out_e, epi_out_c = [], []
        i = 0
        uv_embeds, st_embeds = embeds
        while i < uv_embeds.shape[0]:
            end = min(uv_embeds.shape[0], i+batch_ray_forward)
            result = self.nelf_field([uv_embeds[i:end,...], st_embeds[i:end,...]])
            self.update_sample_near_far(result[0])
            out_e.append(result[0])
            out_c.append(result[1])
            if not without_occ:
                occ_result = self.forward_epi_field(uv_embeds[i:end,...], st_embeds[i:end,...])
                epi_out_e.append(occ_result[1])
                epi_out_c.append(occ_result[2])
            i += batch_ray_forward
        out_e = torch.cat(out_e, dim=0)
        out_c = torch.cat(out_c, dim=0)
        if without_occ:
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
        kwargs['render_params'] = {'batch_ray_forward': 1024}
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs



