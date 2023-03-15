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


def get_line_plane_collision(rays_ori, rays_dir, plane, epsilon=1e-6):
    # https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python
    # for k,v in plane.items():
    #     plane[k] = torch.tensor(v).float().to(rays_ori.device)
    plane_normal, plane_point = plane['normal'], plane['point']
    n_dot_u = (plane_normal[None]*rays_dir).sum(-1)
    if not (n_dot_u.abs()>epsilon).all():
        raise RuntimeError("no intersection or line is within plane")
    w = rays_ori - plane_point
    si = -(plane_normal*w).sum(-1) / n_dot_u
    coord_3d = w + si[:,None] * rays_dir + plane_point[None]
    coord = torch.matmul(plane['proj'], coord_3d.T).T[:,:2]
    # import pdb;pdb.set_trace()
    # t=coord_3d-coord_3d[:1,:]
    # (t*plane_normal[None]).sum(-1)
    # return coord
    return coord_3d[...,:2]


def get_lf_coord(rays_ori, rays_dir, uv_plane, st_plane):
    uv = get_line_plane_collision(rays_ori, rays_dir, uv_plane)
    st = get_line_plane_collision(rays_ori, rays_dir, st_plane)
    return uv, st


@RENDERER.register_module()
class EPINeLFRays(nn.Module):
    def __init__(self, 
                 uv_embedder, 
                 st_embedder, 
                 epi_embedder, 
                 nelf_field, 
                 epi_field, 
                 epi_sample_num,
                 epi_converge_iter, 
                 epi_converge_range,
                 epi_smooth_weight,
                 lf_st_embedder=None, 
                 stop_update=False,
                 consistency_weight=1,
                 background='random',
                 render_params={},):
        super().__init__()
        self.uv_embedder = build_embedder(uv_embedder)
        if lf_st_embedder is None:
            import copy
            lf_st_embedder = copy.deepcopy(st_embedder)
        self.st_embedder = build_embedder(st_embedder)
        self.lf_st_embedder = build_embedder(lf_st_embedder)
        self.epi_embedder = build_embedder(epi_embedder)
        self.nelf_field = build_field(nelf_field)
        self.epi_field = build_field(epi_field)
        self.epi_sample_num = epi_sample_num
        self.epi_converge_iter = epi_converge_iter
        self.epi_converge_range = epi_converge_range
        self.epi_smooth_weight = epi_smooth_weight
        self.consistency_weight = consistency_weight
        self.stop_update = stop_update
        self.background = background
        self.render_params = render_params
        self.fp16_enabled = False
        self.iter = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.percent = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.uv_plane = nn.ParameterDict({
            'normal': nn.Parameter(torch.zeros([3]), requires_grad=False),
            'point': nn.Parameter(torch.zeros([3]), requires_grad=False),
            'proj': nn.Parameter(torch.zeros([3,3]), requires_grad=False),})
        self.st_plane = nn.ParameterDict({
            'normal': nn.Parameter(torch.zeros([3]), requires_grad=False),
            'point': nn.Parameter(torch.zeros([3]), requires_grad=False),
            'proj': nn.Parameter(torch.zeros([3,3]), requires_grad=False),})
        self.center_uv = nn.Parameter(torch.tensor([0.,0.]), requires_grad=False)

        if self.consistency_weight > 0 and stop_update:
            # low-freq init
            self.st_embedder.scale *= 1
            self.epi_embedder.scale *= 1

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
        log_vars['psnr'] = mse2psnr(outputs['rec_mse']).item()
        log_vars['percent'] = self.percent.item()
        outputs.update(dict(loss=loss, log_vars=log_vars))
        outputs['num_samples'] = 1
        return outputs

    @auto_fp16()
    def forward_render(self,
                       rays_ori, rays_dir, rays_color,
                       aug_rays_ori=None, aug_rays_dir=None, 
                       rays_grad=None, max_rays_num=-1, 
                       near=0.0, far=1.0, aug_points=-1, 
                       perturb=True, without_epi=False):
        if len(rays_ori.shape) == 3:
            b, n, _ = rays_ori.shape
            self.b, self.n = b, n
        else:
            b = 1
            n, _ = rays_ori.shape
            self.n = n
            rays_color = rays_color[None]
        if isinstance(near, torch.Tensor):
            self.near = near[0].item()
            self.far = far[0].item()
        else:
            self.near = near
            self.far = far
        rays_ori = rays_ori.reshape([-1,3])
        rays_dir = rays_dir.reshape([-1,3])
        uv, st = get_lf_coord(rays_ori, rays_dir, self.uv_plane, self.st_plane)

        st.requires_grad_(True)
        uv = uv.reshape([-1,2])
        st = st.reshape([-1,2])
        uv_embeds = self.uv_embedder(uv)
        st_embeds = self.lf_st_embedder(st)
        embeds = [uv_embeds, st_embeds]

        if self.training or not max_rays_num>0:
            nelf_epi, rgb = self.nelf_field(embeds)
        else:
            nelf_epi, nelf_rgb, epi, epi_rgb = self.batch_ray_forward(
                uv, st, embeds, max_rays_num, without_epi)
            im_loss = torch.tensor(0., device=nelf_rgb.device)
            if rays_color is not None:
                # im_loss = im2mse(nelf_rgb, rays_color)
                im_loss = im2mse(epi_rgb, rays_color)
            return {'nelf_color_map': nelf_rgb.reshape([b,n,3]), 
                    'epi_color_map': epi_rgb.reshape([b,n,3]), 
                    # 'color_map': nelf_rgb.reshape([b,n,3]), 
                    'color_map': epi_rgb.reshape([b,n,3]), 
                    'nelf_epi_map': nelf_epi.reshape([b,n,1]),
                    'epi_map': epi.reshape([b,n,1]),
                    'rec_mse': im_loss}
        outputs = {}
        outputs['nelf_color_map'] = rgb.reshape([b,n,3])
        outputs['nelf_epi_map'] = nelf_epi.reshape([b,n,1])

        im_loss = im2mse(rgb, rays_color)
        outputs['nelf_rec_loss'] = im_loss

        # sample epi
        self.update_sample_near_far(nelf_epi.detach())
        sample_epi_expanded, sample_epi_embeds = self.get_epi_sample(st.shape[0],perturb)

        # epi field color rec
        aligned_st_embeds = self.sample_st_by_epi(uv, st, self.center_uv, sample_epi_expanded)
        uv_embeds = uv_embeds.expand([self.running_sample_num*2,n,-1])
        # sample_epi_embeds = sample_epi_embeds.expand([n,self.running_sample_num*2,-1]).permute([1,0,2])
        # sample_epi_expanded = sample_epi.expand([n,self.running_sample_num*2]).T[...,None]
        epi_weight, epi, epi_color = self.forward_epi_field(
            uv_embeds, aligned_st_embeds, sample_epi_embeds, sample_epi_expanded)
        # if self.training:
        #     epi_output = torch.ones_like(epi, requires_grad=False)
        #     depi_dst = torch.autograd.grad(outputs=epi, 
        #                                    inputs=st, 
        #                                    grad_outputs=epi_output, 
        #                                    create_graph=True)[0]
        #     outputs['smooth_loss'] = (depi_dst*(-rays_grad).exp()).mean()*self.epi_smooth_weight
        # weight = ((epi_color-rays_color).abs()*10).pow(2).detach()
        # outputs['rec_loss'] = (weight*(epi_color-rays_color).pow(2)).mean()
        outputs['rec_loss'] = im2mse(epi_color, rays_color)
        # outputs['rec_loss'] = (epi_color-rays_color).abs().mean()
        outputs['rec_mse'] = im2mse(epi_color, rays_color)

        # field and epi field consistency
        # 1. on uv,st

        # 2. on aug uv,st
        if aug_rays_dir is not None and self.consistency_weight>0:
            outputs['f_consist_loss'] = ((epi.detach()-nelf_epi)**2).mean()
            self.reset_near_far()
            aug_rays_ori = aug_rays_ori.reshape([-1,3])
            aug_rays_dir = aug_rays_dir.reshape([-1,3])
            aug_uv, aug_st = get_lf_coord(aug_rays_ori, aug_rays_dir, self.uv_plane, self.st_plane)
            aug_uv = aug_uv.reshape([-1,2])
            aug_st = aug_st.reshape([-1,2])
            aug_uv_embeds = self.uv_embedder(aug_uv)
            aug_st_embeds = self.lf_st_embedder(aug_st)
            aug_e, aug_color = self.nelf_field([aug_uv_embeds, aug_st_embeds])
            _, epi_e, epi_color = self.forward_epi_field_uv_st(aug_uv, aug_st)
            outputs['f_consist_loss'] += ((aug_e-epi_e.detach())**2).mean() + \
                                         ((aug_color-epi_color.detach())**2).mean()
            # outputs['f_consist_loss'] += ((aug_e-epi_e)**2).mean() + ((aug_c-epi_c)**2).mean()
            outputs['f_consist_loss'] *= self.consistency_weight
        return outputs

    def sample_st_by_epi(self, ori_uv, ori_st, new_uv, epi):
        with torch.no_grad():
            uv_move = new_uv-ori_uv
            # uv_dist = self.near+epi*(self.far-self.near)
            # st_move = -uv_move/uv_dist*(self.far-uv_dist)
            uv_dist = self.near+epi*(self.far-self.near)
            st_move = -uv_move/uv_dist*(self.far-uv_dist)
            new_st = ori_st+st_move
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
        if self.training and self.background=='random':
            epi_color = epi_color + torch.rand_like(epi_color)*(1-weight.sum(0)) # random background
        if self.background == 'white':
            epi_color = epi_color + 1-weight.sum(0)
        return weight, epi, epi_color

    def forward_epi_field_uv_st(self, uv, st, return_epi_argmax=True):
        n = st.shape[0]
        sample_epi_expanded, sample_epi_embeds = self.get_epi_sample(n)
        aligned_st_embeds = self.sample_st_by_epi(uv, st, self.center_uv, sample_epi_expanded)
        uv_embeds = self.uv_embedder(uv)
        uv_embeds = uv_embeds.expand([self.running_sample_num*2,n,-1])
        return self.forward_epi_field(uv_embeds, 
                                      aligned_st_embeds, 
                                      sample_epi_embeds, 
                                      sample_epi_expanded, 
                                      return_epi_argmax)

    def get_epi_sample(self, n, perturb=False):
        # self.running_sample_num = int(self.epi_sample_num*self.percent/2)
        # with torch.no_grad():
        #     base = torch.linspace(0, 1, self.running_sample_num*2, device=self.iter.device)
        #     sample_epi = base.expand([n, self.running_sample_num*2])

        # self.running_sample_num = int(self.epi_sample_num/2)
        # with torch.no_grad():
        #     base = torch.linspace(0, 1, self.running_sample_num, device=self.iter.device)
        #     avg_sample_epi = base.expand([n, self.running_sample_num])
        #     rand_base = torch.rand([n, self.running_sample_num], device=self.iter.device)
        #     pred_sample_epi = rand_base*(self.sample_far-self.sample_near)+self.sample_near
        #     pred_sample_epi = avg_sample_epi*(self.sample_far-self.sample_near)+self.sample_near
        #     pred_sample_epi = pred_sample_epi.expand([n, self.running_sample_num])
        #     sample_epi = torch.cat([avg_sample_epi, pred_sample_epi], dim=-1)
        #     sample_epi = sample_epi.sort(dim=-1).values

        # self.running_sample_num = int(self.epi_sample_num*self.percent/2)
        # with torch.no_grad():
        #     base = torch.linspace(0, 1, self.running_sample_num, device=self.iter.device)
        #     avg_sample_epi = base.expand([n, self.running_sample_num])
        #     rand_base = torch.rand([n, self.running_sample_num], device=self.iter.device)
        #     pred_sample_epi = rand_base*(self.sample_far-self.sample_near)+self.sample_near
        #     pred_sample_epi = avg_sample_epi*(self.sample_far-self.sample_near)+self.sample_near
        #     pred_sample_epi = pred_sample_epi.expand([n, self.running_sample_num])
        #     sample_epi = torch.cat([avg_sample_epi, pred_sample_epi], dim=-1)
        #     sample_epi = sample_epi.sort(dim=-1).values

        self.running_sample_num = self.epi_sample_num//2
        with torch.no_grad():
            base = torch.linspace(0, 1, self.running_sample_num*2, device=self.iter.device)
            sample_epi = base*(self.sample_far-self.sample_near)+self.sample_near
            sample_epi = sample_epi.expand([n, self.running_sample_num*2])

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
        near = 0
        far = 1
        with torch.no_grad():
            if self.iter == self.epi_converge_iter and self.stop_update:
                self.consistency_weight = 0
                self.st_embedder.scale /= 1
                self.epi_embedder.scale /= 1
            if self.iter > self.epi_converge_iter:
                percent = self.epi_converge_range
            else:
                progress = self.iter/self.epi_converge_iter
                percent = 1*(1-progress) + (self.epi_converge_range*progress)
            self.percent *= 0 # it is a parameter
            self.percent += percent
            self.sample_near = nelf_epi-percent*(far-near)/2
            small_mask = self.sample_near<near
            near_ones = torch.ones_like(self.sample_near)
            self.sample_near[small_mask] = near_ones[small_mask]*near
            self.sample_near = self.sample_near*percent + near_ones*(1-percent)*near

            self.sample_far = nelf_epi+percent*(far-near)/2
            large_mask = self.sample_far>far
            far_ones = torch.ones_like(self.sample_far)
            self.sample_far[large_mask] = far_ones[large_mask]*far
            self.sample_far = self.sample_far*percent + far_ones*(1-percent)*far

    def reset_near_far(self):
        self.sample_near = 0
        self.sample_far = 1

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
        outputs['rec_loss'] = outputs['rec_mse']
        outputs = self._parse_outputs(outputs)
        return outputs



