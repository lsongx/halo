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
    coord = torch.matmul(coord_3d, plane['proj'].T)[...,:2]
    return coord_3d, coord
    # return coord_3d, coord_3d[...,:2]

def proj_coord_3d(coord_3d, plane):
    return torch.matmul(coord_3d, plane['proj'].T)[...,:2]

def get_lf_coord(rays_ori, rays_dir, uv_plane, st_plane):
    uv_3d, uv = get_line_plane_collision(rays_ori, rays_dir, uv_plane)
    st_3d, st = get_line_plane_collision(rays_ori, rays_dir, st_plane)
    return uv_3d, st_3d, uv, st


@RENDERER.register_module()
class EPINeLF(nn.Module):
    def __init__(self, 
                 uv_embedder, 
                 st_embedder, 
                 epi_embedder, 
                 epi_field, 
                 nelf_field, 
                 epi_near, 
                 epi_far,
                 epi_sample_num,
                 epi_converge_iter, 
                 epi_converge_range,
                 render_params,):
        super().__init__()
        self.uv_embedder = build_embedder(uv_embedder)
        self.st_embedder = build_embedder(st_embedder)
        self.epi_embedder = build_embedder(epi_embedder)
        self.epi_field = build_field(epi_field)
        self.nelf_field = build_field(nelf_field)
        self.epi_sample_num = epi_sample_num
        self.epi_converge_iter = epi_converge_iter
        self.epi_converge_range = epi_converge_range
        self.epi_near = nn.Parameter(torch.tensor(epi_near), requires_grad=False)
        self.epi_far = nn.Parameter(torch.tensor(epi_far), requires_grad=False)
        self.render_params = render_params
        self.fp16_enabled = False
        self.iter = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.max_moving_dis = nn.Parameter(torch.ones([2]), requires_grad=False)
        self.min_moving_dis = nn.Parameter(torch.ones([2]), requires_grad=False)
        self.uv_plane = nn.ParameterDict({
            'normal': nn.Parameter(torch.zeros([3]), requires_grad=False),
            'point': nn.Parameter(torch.zeros([3]), requires_grad=False),
            'proj': nn.Parameter(torch.zeros([3,3]), requires_grad=False),})
        self.st_plane = nn.ParameterDict({
            'normal': nn.Parameter(torch.zeros([3]), requires_grad=False),
            'point': nn.Parameter(torch.zeros([3]), requires_grad=False),
            'proj': nn.Parameter(torch.zeros([3,3]), requires_grad=False),})

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
                       rays_ori, rays_dir, rays_color, 
                       near=0.0, far=1.0,
                       aug_rays_ori=None, aug_rays_dir=None, 
                       sample_rays_ori=None, sample_rays_dir=None, 
                       max_rays_num=-1, aug_points=-1, 
                       perturb=True, without_occ=False):
        n = rays_dir.shape[0]
        uv_3d, st_3d, uv, st = get_lf_coord(rays_ori, rays_dir, self.uv_plane, self.st_plane)
        uv_embeds = self.uv_embedder(uv.reshape([-1,2]))
        st_embeds = self.st_embedder(st.reshape([-1,2]))
        embeds = [uv_embeds, st_embeds]

        if self.training or max_rays_num<=0:
            epi_dir, color_code, rgb = self.nelf_field(embeds)
            epi_dir = epi_dir*(self.epi_far-self.epi_near)+self.epi_near
        else:
            epi_dir, rgb, occ_e, occ_rgb = self.batch_ray_forward(embeds, max_rays_num, without_occ)
            epi_dir = epi_dir*(self.epi_far-self.epi_near)+self.epi_near
            im_loss = torch.tensor(0., device=rgb.device)
            if rays_color is not None:
                im_loss = im2mse(rgb, rays_color)
            return {'color_map': rgb, 
                    'occ_color_map': occ_rgb, 
                    'epi_map': epi_dir,
                    'occ_epi_map': occ_e,
                    'rec_loss': im_loss}
        outputs = {}
        outputs['color_map'] = rgb
        outputs['epi_map'] = epi_dir

        im_loss = im2mse(rgb, rays_color)
        outputs['nelf_rec_loss'] = im_loss

        # sample epi
        self.update_sample_near_far(epi_dir)
        epi_expanded, epi_embeds = self.get_epi(perturb)

        # occ field color rec
        uv_embeds = uv_embeds.expand([self.epi_sample_num,n,-1])
        st_embeds = st_embeds.expand([self.epi_sample_num,n,-1])
        occ_weight, occ_epi, occ_code, occ_rgb = self.forward_epi_field(
            uv_embeds, st_embeds, epi_embeds, epi_expanded)

        # outputs['occ_rec_loss'] = im2mse(occ_rgb, rays_color)
        outputs['rec_loss'] = im2mse(occ_rgb, rays_color)
        # outputs['occ_weight_loss'] = ((occ_weight.sum(0)-1)**2).mean()

        # occ field consistency
        # 1. sample other rays; rays on other uv grid
        # 2. consistency loss (l2)
        if sample_rays_dir is not None:
            sample_uv_3d, sample_st_3d, sample_uv, sample_st = get_lf_coord(
                sample_rays_ori, sample_rays_dir, self.uv_plane, self.st_plane)
            sample_uv_embeds, sample_st_embeds = self.sample_st_by_epi(
                uv_3d, st_3d, sample_uv_3d, epi_expanded)
            _, _, _, s_rgb = self.forward_epi_field(
                sample_uv_embeds, sample_st_embeds, epi_embeds, epi_expanded)
            outputs['occ_aug_rec_loss'] = im2mse(s_rgb, rays_color)

        # 3. sample other rays; rays on random uv (aug uv)
        # 4. consistency loss (l2)
        if aug_rays_dir is not None:
            aug_uv_3d, aug_st_3d, aug_uv, aug_st = get_lf_coord(
                aug_rays_ori, aug_rays_dir, self.uv_plane, self.st_plane)
            aug_uv_embeds, aug_st_embeds = self.sample_st_by_epi(
                uv_3d, st_3d, aug_uv_3d, epi_expanded)
            _, _, _, s_rgb = self.forward_epi_field(
                aug_uv_embeds, aug_st_embeds, epi_embeds, epi_expanded)
            outputs['occ_aug_rec_loss'] += im2mse(s_rgb, rays_color)

        # field and occ field consistency
        # 1. on uv,st
        outputs['f_consist_loss'] = ((occ_epi.detach()-epi_dir)**2).mean()
        # outputs['f_consist_loss'] = ((occ_epi-epi_dir)**2).mean() + \
        #                             ((out_occ_color_code-color_code)**2).mean()

        # 2. on aug uv,st
        aug_uv_embeds, aug_st_embeds = aug_uv_embeds[0], aug_st_embeds[0]
        aug_e, aug_c, aug_rgb = self.nelf_field([aug_uv_embeds, aug_st_embeds],)
        aug_e = aug_e*(self.epi_far-self.epi_near)+self.epi_near
        with torch.no_grad():
            _, occ_e, occ_c, occ_rgb = self.forward_epi_field(aug_uv_embeds, aug_st_embeds)
        outputs['f_consist_loss'] += ((aug_e-occ_e)**2).mean() + \
                                     ((aug_rgb-occ_rgb)**2).mean()
                                    #  ((aug_c-occ_c.detach())**2).mean()
        # outputs['f_consist_loss'] += ((aug_e-occ_e)**2).mean() + ((aug_c-occ_c)**2).mean()
        return outputs

    def sample_st_by_epi(self, ori_uv, ori_st, new_uv, epi):
        with torch.no_grad():
            uv_move = new_uv-ori_uv
            st_move = -uv_move[None]*(1-epi)/epi
            new_st = ori_st[None]+st_move
            new_uv = new_uv.expand_as(new_st)
            new_st = proj_coord_3d(new_st, self.st_plane)
            new_uv = proj_coord_3d(new_uv, self.st_plane)
            new_uv_embeds = self.uv_embedder(new_uv)
            new_st_embeds = self.st_embedder(new_st)
        return new_uv_embeds, new_st_embeds

    def forward_epi_field(self, uv_embeds, st_embeds, epi_embeds=None, epi_expanded=None):
        if epi_embeds is None: # forward with uv-st
            n = uv_embeds.shape[0]
            uv_embeds = uv_embeds.expand([self.epi_sample_num,n,-1])
            st_embeds = st_embeds.expand([self.epi_sample_num,n,-1])
            epi_expanded, epi_embeds = self.get_epi()
        # distance = epi_expanded[1:,:,:]-epi_expanded[:-1,:,:]
        # distance = torch.cat([distance, 1e8*torch.ones_like(distance[:1,...])], dim=0)
        # distance = torch.cat([distance, torch.ones_like(distance[:1,...])], dim=0)
        distance = 1/self.epi_sample_num
        alpha, occ_color_code = self.epi_field(uv_embeds, st_embeds, epi_embeds, distance)

        cp = torch.cumprod(1-alpha, 0)
        cp = torch.roll(cp, 1, 0)
        cp[0] = 1.0
        weight = cp*alpha
        occ_code = (weight*occ_color_code).sum(0)
        # occ_epi = (weight*epi_expanded).sum(0)
        occ_epi = epi_expanded.gather(dim=0,index=weight.argmax(0)[None])[0]
        # occ_epi = torch.clamp(occ_epi, self.epi_near, self.epi_far)
        # occ_rgb = self.nelf_field.forward_with_color_code(occ_code, uv_embeds[0],)
        occ_rgb = self.nelf_field.forward_with_color_code(occ_color_code, uv_embeds)
        occ_rgb = (weight*occ_rgb).sum(0) 
        occ_rgb = occ_rgb + torch.rand_like(occ_rgb)*(1-weight.sum(0)) # random background
        return weight, occ_epi, occ_code, occ_rgb

    def get_epi(self, perturb=False):
        with torch.no_grad():
            base = torch.linspace(0, 1, self.epi_sample_num, device=self.iter.device)
            epi = base*(self.sample_far-self.sample_near)
            epi += self.sample_near
            epi = epi.T[..., None] # [n_epi_sample, n_ray, 1]
            if perturb:
                # Gets intervals
                mid_points = 0.5 * (epi[1:] + epi[:-1])
                upper = torch.cat([mid_points, epi[-1:]], 0)
                lower = torch.cat([epi[:1], mid_points], 0)
                # stratified samples in those intervals
                t_rand = torch.rand(epi.shape, device=epi.device)
                epi = lower + (upper-lower)*t_rand            
            epi_embeds = self.epi_embedder(epi)
        return epi, epi_embeds

    def update_sample_near_far(self, epi_dir):
        # if self.iter % 10000 != 0 and self.training:
        #     # during training
        #     return
        with torch.no_grad():
            progress = min(self.iter, self.epi_converge_iter)/self.epi_converge_iter
            percent = 1*(1-progress) + self.epi_converge_range*progress
            self.sample_near = epi_dir-percent*(self.epi_far-self.epi_near)/2
            small_mask = self.sample_near<self.epi_near
            near_ones = torch.ones_like(self.sample_near)
            self.sample_near[small_mask] = near_ones[small_mask]*self.epi_near
            self.sample_near = self.sample_near*progress + near_ones*(1-progress)*self.epi_near

            self.sample_far = epi_dir+percent*(self.epi_far-self.epi_near)/2
            large_mask = self.sample_far>self.epi_far
            far_ones = torch.ones_like(self.sample_far)
            self.sample_far[large_mask] = far_ones[large_mask]*self.epi_far
            self.sample_far = self.sample_far*progress + far_ones*(1-progress)*self.epi_far

    def batch_ray_forward(self, embeds, max_rays_num, without_occ):
        out_e, out_r = [], []
        occ_out_e, occ_out_r = [], []
        i = 0
        uv_embeds, st_embeds = embeds
        while i < uv_embeds.shape[0]:
            end = min(uv_embeds.shape[0], i+max_rays_num)
            result = self.nelf_field([uv_embeds[i:end,...], st_embeds[i:end,...]],)
            self.update_sample_near_far(result[0])
            out_e.append(result[0])
            out_r.append(result[2])
            if without_occ:
                occ_result = result
            else:
                occ_result = self.forward_epi_field(uv_embeds[i:end,...], st_embeds[i:end,...])[1:]
            occ_out_e.append(occ_result[0])
            occ_out_r.append(occ_result[2])
            i += max_rays_num
        out_e = torch.cat(out_e, dim=0)
        out_r = torch.cat(out_r, dim=0)
        occ_out_e = torch.cat(occ_out_e, dim=0)
        occ_out_r = torch.cat(occ_out_r, dim=0)
        return out_e, out_r, occ_out_e, occ_out_r

    def train_step(self, data, optimizer, **kwargs):
        for k, v in data.items():
            if v.shape[0] == 1:
                data[k] = v[0] # batch size = 1
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        kwargs['render_params'] = {'max_rays_num': 1024}
        outputs = self.train_step(data, optimizer, **kwargs)
        return outputs



