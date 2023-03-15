from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmcv.runner import auto_fp16, force_fp32
from vcnerf.core import im2mse, mse2psnr, raw2outputs, SamplePDF

from ..builder import RENDERER, build_embedder, build_field


def get_sphere_coord_with_rad(rays_ori, rays_dir, rad):
    # http://www.ambrsoft.com/TrigoCalc/Sphere/SpherLineIntersection_.htm
    # a = (x2 − x1)2 + (y2 − y1)2 + (z2 − z1)2
    # b = − 2[(x2 − x1)(xc − x1) + (y2 − y1)(yc − y1) + (z2 − z1)(zc − z1)]
    # c = (xc − x1)2 + (yc − y1)2 + (zc − z1)2 − r2
    # t = (-b±√(b^2-4ac))/2a

    # (x2-x1, y2-y1, z2-z1): rays_dir
    # (x1, y1, z1): rays_ori

    a = (rays_dir**2).sum(dim=1)
    b = -2 * (rays_dir*(-rays_ori)).sum(dim=1)
    c = (rays_ori**2).sum(dim=1) - rad
    delta = (b**2-4*a*c) > 0
    if not torch.all(delta):
        raise RuntimeError('not all deltas are positive')

    t1 = (-b+torch.sqrt(b**2-4*a*c)) / (2*a)
    t2 = (-b-torch.sqrt(b**2-4*a*c)) / (2*a)

    point1 = rays_ori+rays_dir*t1[:,None]
    point2 = rays_ori+rays_dir*t2[:,None]

    point1_dist = (point1-rays_ori).abs().sum()
    point2_dist = (point2-rays_ori).abs().sum()
    if point1_dist > point2_dist:
        point_selected = point2
    else:
        point_selected = point1
    # always use the nearest one, convert from points to coord

    # (x,y) as coord
    sphere_coord = point_selected[:,:2]
    valid_mask = point_selected[:,2]>0
    return sphere_coord, valid_mask


def get_sphere_coord(rays_ori, rays_dir, rad0, rad1):
    st, st_valid_mask = get_sphere_coord_with_rad(rays_ori, rays_dir, rad0)
    uv, uv_valid_mask = get_sphere_coord_with_rad(rays_ori, rays_dir, rad1)
    valid_mask = st_valid_mask & uv_valid_mask
    return uv, st, valid_mask


@RENDERER.register_module()
class NeLFNeRF(nn.Module):
    def __init__(self, 
                 xyz_embedder, 
                 lf_embedder,
                 coarse_field, 
                 nelf_field,
                 rad0, rad1,
                 render_params={},
                 dir_embedder=None,):
        super().__init__()
        self.xyz_embedder = build_embedder(xyz_embedder)
        self.lf_embedder = build_embedder(lf_embedder)
        self.dir_embedder = build_embedder(dir_embedder)
        self.coarse_field = build_field(coarse_field)
        self.nelf_field = build_field(nelf_field)
        self.render_params = render_params
        self.rad0 = rad0
        self.rad1 = rad1

        self.fp16_enabled = False
        self.iter = nn.Parameter(torch.tensor(0), requires_grad=False)

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
                       rays_ori, rays_dir, aug_rays_ori, aug_rays_dir,
                       rays_color, ndc_rays_ori, ndc_rays_dir, # loader output
                       n_samples, perturb, alpha_noise_std, inv_depth, # render param
                       use_dirs, max_rays_num, ndc=False, near=0.0, far=1.0, background=False):

        render_args = {
            'n_samples': n_samples, 'perturb': perturb, 'alpha_noise_std': alpha_noise_std, 
            'inv_depth': inv_depth, 'use_dirs': use_dirs, 'max_rays_num': max_rays_num, 
            'ndc': ndc, 'near': near, 'far': far, 'background': background,}

        nerf_outputs = self.forward_nerf(rays_ori, rays_dir, ndc_rays_ori, ndc_rays_dir, **render_args)

        outputs = {
            'nerf_color_map': nerf_outputs['color_map'],
            'nerf_max_occ_map': nerf_outputs['max_occ_map']}
        im_loss = im2mse(nerf_outputs['color_map'], rays_color)
        outputs['nerf_rec_loss'] = im_loss

        uv, st, valid_mask = get_sphere_coord(rays_ori, rays_dir, self.rad0, self.rad1)
        uv_embeds = self.lf_embedder(uv)
        st_embeds = self.lf_embedder(st)
        max_occ, _, rgb = self.nelf_field([uv_embeds, st_embeds], near=near, far=far)
        outputs['rec_loss'] = im2mse(rgb, rays_color, valid_mask)
        outputs['color_map'] = rgb
        outputs['max_occ_map'] = max_occ

        if self.training:
            aug_uv, aug_st, valid_mask = get_sphere_coord(aug_rays_ori, aug_rays_dir, self.rad0, self.rad1)
            aug_uv_embeds = self.lf_embedder(aug_uv)
            aug_st_embeds = self.lf_embedder(aug_st)
            aug_max_occ, _, aug_rgb = self.nelf_field([aug_uv_embeds, aug_st_embeds], near=near, far=far)

            render_args.update({'perturb': False, 'alpha_noise_std': 0})
            aug_outputs = self.forward_nerf(rays_ori, rays_dir, ndc_rays_ori, ndc_rays_dir, **render_args)
            outputs['consist_loss'] = im2mse(aug_rgb, aug_outputs['color_map'].detach()) + \
                                    im2mse(aug_max_occ, aug_outputs['max_occ_map'].detach()) + \
                                    im2mse(max_occ, nerf_outputs['max_occ_map'].detach())
            # outputs['consist_loss'] *= 0

        return outputs

    @auto_fp16()
    def forward_nerf(self,
                     rays_ori, rays_dir, ndc_rays_ori, ndc_rays_dir, # loader output
                     n_samples, alpha_noise_std, inv_depth, # render param
                     use_dirs, max_rays_num, perturb=False, ndc=False, 
                     near=0.0, far=1.0, background=False):
        # near, far: [B] or [B, H, W]
        near = near * torch.ones(rays_ori.shape[:-1], dtype=torch.float32, device=rays_ori.device)
        far = far * torch.ones(rays_ori.shape[:-1], dtype=torch.float32, device=rays_ori.device)
        t_vals = torch.linspace(0, 1, n_samples, dtype=torch.float32, device=rays_ori.device)
        if not inv_depth:
            # z_vals: [B, n_samples] or [B, H, W, n_samples]
            z_vals = near[..., None] * (1 - t_vals) + far[..., None] * t_vals
        else:
            z_vals = 1/(1 / near[..., None] * (1 - t_vals) + 1 / far[..., None] * t_vals)
        
        # Perturbs points coordinates
        if perturb:
            # Gets intervals
            mid_points = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mid_points, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mid_points], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, dtype=torch.float32, device=rays_ori.device)
            z_vals = lower + (upper - lower) * t_rand
        
        if use_dirs:
            directions = F.normalize(rays_dir, p=2, dim=-1)
        else:
            directions = None
        
        if ndc:
            # points in space to evaluate model at
            points = ndc_rays_ori[..., None, :] + ndc_rays_dir[..., None, :] * \
                z_vals[..., :, None]  # [B, n_samples, 3] or [B, H, W, n_samples, 3]
        else:
            # points in space to evaluate model at
            points = rays_ori[..., None, :] + rays_dir[..., None, :] * \
                z_vals[..., :, None]  # [B, n_samples, 3] or [B, H, W, n_samples, 3]

        # Evaluates the model at the points
        alphas, colors = self.forward_batchified(points, 
                                                 directions, 
                                                 max_rays_num=max_rays_num)
        nerf_outputs = raw2outputs(alphas, 
                                   colors, 
                                   z_vals, 
                                   ndc_rays_dir if ndc else rays_dir,
                                   alpha_noise_std,
                                   background)
        return nerf_outputs

    @auto_fp16()
    def forward_batchified(self, 
                           points, 
                           directions, 
                           max_rays_num,):
        assert points.shape[0] == directions.shape[0], (
            f'points: {points.shape}, directions: {directions.shape}')
        nb_rays = points.shape[0]
        if nb_rays <= max_rays_num:
            return self.forward_points(points, directions)
        else:
            outputs = []
            start = 0
            end = max_rays_num
            while start < nb_rays:
                assert start < end, 'start >= end ({:d}, {:d})'.format(start, end)
                output = self.forward_points(points[start: end, ...], 
                                             directions[start: end, ...], )
                outputs.append(output)
                start += max_rays_num
                end = min(end + max_rays_num, nb_rays)
            
            alphas_colors = []
            for i, out in enumerate(zip(*outputs)):
                if out[0] is not None:
                    out = torch.cat(out, dim=0)
                else:
                    out = None
                alphas_colors.append(out)
            return alphas_colors

    @auto_fp16(apply_to=('points',))
    def forward_points(self, points, directions=None,):
        shape = tuple(points.shape[:-1])  # [B, n_points]
        # [B, 3] -> [B, n_points, 3]
        directions = directions[..., None, :].expand_as(points)
        
        points = points.reshape((-1, 3))
        directions = directions.reshape((-1, 3))

        xyz_embeds = self.xyz_embedder(points)
        if self.dir_embedder is None:
            dir_embeds = None
        else:
            assert self.dir_embedder is not None
            dir_embeds = self.dir_embedder(directions)
        coarse_alphas, coarse_colors = self.coarse_field(xyz_embeds, dir_embeds)
        # [B, n_points, 1/3]
        coarse_alphas = coarse_alphas.reshape(shape + (1,))
        coarse_colors = coarse_colors.reshape(shape + (3,))

        return coarse_alphas, coarse_colors

    def train_step(self, data, optimizer, **kwargs):
        for k, v in data.items():
            if v.shape[0] == 1:
                data[k] = v[0] # batch size = 1
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        return self.train_step(data, optimizer, **kwargs)

