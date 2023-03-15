from collections import OrderedDict
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmcv.runner import auto_fp16, force_fp32
from vcnerf.utils.logger import get_root_logger
from vcnerf.core import im2mse, mse2psnr, raw2outputs, SamplePDF

from ..builder import RENDERER, build_embedder, build_field


def get_lf_coord(rays_ori, rays_dir):
    norm_rays_dir = F.normalize(rays_dir, p=2, dim=-1)
    return rays_ori, norm_rays_dir


def load_pth(path, contain_keys=''):
    pth = torch.load(path, 'cpu')
    if 'state_dict' in pth.keys():
        pth = pth['state_dict']
    if 'module' in list(pth.keys())[0]:
        pth = {k.replace('module.', ''): v for k, v in pth.items()}
    if contain_keys:
        pth = {k.replace(f'{contain_keys}.', ''): v 
               for k, v in pth.items() if contain_keys in k}
    return pth


@RENDERER.register_module()
class NeLFNeRF360LFHF(nn.Module):
    def __init__(self, 
                 xyz_embedder, 
                 uv_embedder,
                 st_embedder,
                 coarse_field, 
                 nelf_field,
                 lf_nerf_field, 
                 lf_xyz_embedder,
                 rec_loss_weight=1,
                 consist_loss_weight=1,
                 hf_load_lf=False,
                 empty_start_iter=5e3,
                 empty_loss_weight=0.1,
                 nelf_pretrain=None,
                 nerf_pretrain=None,
                 render_params={},
                 dir_embedder=None,):
        super().__init__()
        self.logger = get_root_logger()
        self.xyz_embedder = build_embedder(xyz_embedder)
        self.uv_embedder = build_embedder(uv_embedder)
        self.st_embedder = build_embedder(st_embedder)
        self.dir_embedder = build_embedder(dir_embedder)
        self.coarse_field = build_field(coarse_field)
        self.lf_nerf_field = build_field(lf_nerf_field)
        self.lf_xyz_embedder = build_embedder(lf_xyz_embedder)
        self.nelf_field = build_field(nelf_field)
        if nelf_pretrain is not None and os.path.isfile(nelf_pretrain):
            self.nelf_field.load_state_dict(load_pth(nelf_pretrain, 'nelf_field'))
            self.logger.info(f'{nelf_pretrain} loaded')
        if nerf_pretrain is not None and os.path.isfile(nerf_pretrain):
            checkpoint = load_pth(nerf_pretrain, 'coarse_field')
            self.lf_nerf_field.load_state_dict(checkpoint)
            self.logger.info(f'{nerf_pretrain} loaded')
            if hf_load_lf:
                for k in list(checkpoint.keys()):
                    if 'layers.fc0' in k or 'layers.fc4' in k:
                        checkpoint.pop(k)
                self.coarse_field.load_state_dict(checkpoint, strict=False)
                self.logger.info(f'{nerf_pretrain} loaded for hf nerf')
        self.empty_start_iter = empty_start_iter
        self.render_params = render_params
        self.rec_loss_weight = rec_loss_weight
        self.empty_loss_weight = empty_loss_weight
        self.consist_loss_weight = consist_loss_weight
        self.sample_pdf = SamplePDF()

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
                       rays_ori, rays_dir, rays_color, # loader output
                       n_samples, perturb, alpha_noise_std, inv_depth, # render param
                       use_dirs, max_rays_num, aug_rays_ori=None, aug_rays_dir=None, 
                       ndc=False, near=0.0, far=1.0, z_by_nelf=-1, background=False):

        outputs = {}
        uv, st = get_lf_coord(rays_ori, rays_dir)
        uv_embeds = self.uv_embedder(uv)
        st_embeds = self.st_embedder(st)
        if self.consist_loss_weight==0 and z_by_nelf>0:
            with torch.no_grad():
                max_occ, occ = self.nelf_field([uv_embeds, st_embeds], near=near, far=far)
        else:
            max_occ, occ = self.nelf_field([uv_embeds, st_embeds], near=near, far=far)
        occ = occ.mean(dim=-1)
        outputs['nelf_color_map'] = occ
        outputs['nelf_max_occ_map'] = max_occ
        # outputs['depth_map'] = max_occ
        # outputs['color_map'] = rgb

        # outputs['nelf_rec_loss'] = im2mse(rgb, rays_color)*self.consist_loss_weight
        # outputs['rec_loss'] = im2mse(rgb, rays_color)

        render_args = {
            'field': self.coarse_field, 'embedder': self.xyz_embedder,
            'n_samples': n_samples, 'perturb': perturb, 
            'alpha_noise_std': alpha_noise_std, 'inv_depth': inv_depth, 
            'use_dirs': use_dirs, 'max_rays_num': max_rays_num, 
            'ndc': ndc, 'near': near, 'far': far, 
            # 'z_by_nelf': -1, 'max_occ': max_occ.detach(), 'background': background,}
            'z_by_nelf': z_by_nelf, 'max_occ': max_occ.detach(), 'background': background,}

        # with torch.no_grad():
        #     lf_render_args = {**render_args, 'perturb': False, 'alpha_noise_std': 0, 
        #                         'n_samples': 128, 'field': self.lf_nerf_field, 
        #                         'embedder': self.lf_xyz_embedder}
        #     lf_nerf_outputs = self.forward_nerf(aug_rays_ori, aug_rays_dir, **lf_render_args)
        #     weights = lf_nerf_outputs['weights'][...,1:-1]
        #     z_vals = lf_nerf_outputs['z_vals']
        #     z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        #     z_samples = self.sample_pdf(z_vals_mid, weights, n_samples, not perturb)
        #     z_vals_fine, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        #     render_args['given_z'] = z_vals_fine
        #     empty_mask = lf_nerf_outputs['acc_map']<0.01

        nerf_outputs = self.forward_nerf(rays_ori, rays_dir, **render_args)
        outputs.update({'nerf_color_map': nerf_outputs['color_map'],
                        'nerf_max_occ_map': nerf_outputs['max_occ_map'],
                        'nerf_depth_map': nerf_outputs['depth_map']})

        im_loss = im2mse(nerf_outputs['color_map'], rays_color)
        # outputs['nerf_rec_loss'] = im_loss
        outputs['rec_loss'] = im_loss*self.rec_loss_weight
        outputs['color_map'] = nerf_outputs['color_map']
        # outputs['depth_map'] = nerf_outputs['depth_map']

        # if self.consist_loss_weight>0:
        #     outputs['consist_loss'] = im2mse(max_occ/far, nerf_outputs['max_occ_map'].detach()/far)*self.consist_loss_weight
        #     outputs['nelf_occ_loss'] = im2mse(occ, nerf_outputs['acc_map'].detach())*self.consist_loss_weight

        if self.training and z_by_nelf>0:
            # if (self.iter<5e3 or self.iter>6e4 or random.random()>0.8) and im_loss.requires_grad:
            if (self.iter<self.empty_start_iter or self.iter>9e4) and im_loss.requires_grad:
                outputs['empty_consist_loss'] = torch.zeros_like(outputs['rec_loss'])
            else:
                # outputs['empty_consist_loss'] = (nerf_outputs['acc_map'][empty_mask]).abs().mean()*self.rec_loss_weight*0.1
                render_args['z_by_nelf'] = -1
                # nerf_outputs_even = self.forward_nerf(rays_ori, rays_dir, **render_args)
                # outputs['even_rec_loss'] = im2mse(nerf_outputs_even['color_map'], rays_color)*self.rec_loss_weight

                aug_outputs = self.forward_nerf(aug_rays_ori, aug_rays_dir, **render_args)
                with torch.no_grad():
                    render_args.update({'perturb': False, 'alpha_noise_std': 0, 'n_samples': 128,
                                        'field': self.lf_nerf_field, 'embedder': self.lf_xyz_embedder})
                    lf_nerf_outputs = self.forward_nerf(aug_rays_ori, aug_rays_dir, **render_args)
                    # empty_mask = (lf_nerf_outputs['acc_map']<0.01) & (lf_nerf_outputs['depth_map']>0.1)
                    empty_mask = lf_nerf_outputs['acc_map']<0.01
                    outputs.update({'lf_nerf_color_map': lf_nerf_outputs['color_map'],
                                    'lf_nerf_acc_map': lf_nerf_outputs['acc_map'],
                                    'lf_nerf_empty_mask': empty_mask,
                                    'lf_nerf_depth_map': lf_nerf_outputs['depth_map']})
                outputs['empty_consist_loss'] = (aug_outputs['acc_map'][empty_mask]).abs().mean()*self.empty_loss_weight
                # outputs['empty_consist_loss'] = (aug_outputs['acc_map'][empty_mask]).abs().mean()

        # if self.training and self.consist_loss_weight>0:
        #     aug_uv, aug_st = get_lf_coord(aug_rays_ori, aug_rays_dir)
        #     aug_uv_embeds = self.uv_embedder(aug_uv)
        #     aug_st_embeds = self.st_embedder(aug_st)
        #     aug_max_occ, aug_occ = self.nelf_field([aug_uv_embeds, aug_st_embeds], near=near, far=far)
        #     aug_occ = aug_occ.mean(-1)

        #     render_args.update({'perturb': False, 'alpha_noise_std': 0})
        #     with torch.no_grad():
        #         aug_outputs = self.forward_nerf(aug_rays_ori, aug_rays_dir, **render_args)
            
        #     outputs['consist_loss'] += im2mse(aug_max_occ/far, aug_outputs['max_occ_map'].detach()/far)*self.consist_loss_weight 
        #     outputs['nelf_occ_loss'] += im2mse(aug_occ, aug_outputs['acc_map'].detach())*self.consist_loss_weight

        #     outputs['nerf_aug_color_map'] = aug_outputs['color_map'].detach()
        #     outputs['nelf_aug_color_map'] = aug_occ
        #     outputs['nerf_aug_max_occ'] = aug_outputs['max_occ_map'].detach()
        #     outputs['nelf_aug_max_occ'] = aug_max_occ.detach()

        return outputs

    @auto_fp16()
    def forward_nerf(self, 
                     rays_ori, rays_dir, # loader output
                     field, embedder,
                     n_samples, alpha_noise_std, inv_depth, # render param
                     use_dirs, max_rays_num, 
                     perturb=False, ndc=False, 
                     near=0.0, far=1.0, given_z=None,
                     z_by_nelf=-1, max_occ=None, background=False):
        range = far - near
        # near, far: [B] or [B, H, W]
        near = near * torch.ones([*rays_ori.shape[:-1],1], dtype=rays_ori.dtype, device=rays_ori.device)
        far = far * torch.ones([*rays_ori.shape[:-1],1], dtype=rays_ori.dtype, device=rays_ori.device)
        t_vals = torch.linspace(0, 1, n_samples, dtype=rays_ori.dtype, device=rays_ori.device)

        if z_by_nelf > 0:
            near_new = max_occ-range*z_by_nelf
            near_new[near_new<near] = near[near_new<near]
            near = near_new
            far_new = max_occ+range*z_by_nelf
            far_new[far_new<far] = far[far_new<far]
            far = far_new

        if not inv_depth:
            # z_vals: [B, n_samples] or [B, H, W, n_samples]
            z_vals = near * (1 - t_vals) + far * t_vals
        else:
            z_vals = 1/(1 / near * (1 - t_vals) + 1 / far * t_vals)

        # Perturbs points coordinates
        if perturb:
            # Gets intervals
            mid_points = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mid_points, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mid_points], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, dtype=rays_ori.dtype, device=rays_ori.device)
            z_vals = lower + (upper - lower) * t_rand
        
        if given_z is not None:
            z_vals = given_z

        if use_dirs:
            directions = F.normalize(rays_dir, p=2, dim=-1)
        else:
            directions = None
        
        # points in space to evaluate model at
        points = rays_ori[..., None, :] + rays_dir[..., None, :] * \
                 z_vals[..., :, None]  # [B, n_samples, 3] or [B, H, W, n_samples, 3]

        # Evaluates the model at the points
        alphas, colors = self.forward_batchified(field,
                                                 embedder,
                                                 points, 
                                                 directions, 
                                                 max_rays_num=max_rays_num)
        nerf_outputs = raw2outputs(alphas, 
                                   colors, 
                                   z_vals, 
                                   rays_dir,
                                   alpha_noise_std,
                                   background)
        nerf_outputs['z_vals'] = z_vals
        return nerf_outputs

    @auto_fp16()
    def forward_batchified(self, 
                           field,
                           embedder,
                           points, 
                           directions, 
                           max_rays_num,):
        assert points.shape[0] == directions.shape[0], (
            f'points: {points.shape}, directions: {directions.shape}')
        nb_rays = points.shape[0]
        if nb_rays <= max_rays_num:
            return self.forward_points(field, embedder, points, directions)
        else:
            outputs = []
            start = 0
            end = max_rays_num
            while start < nb_rays:
                assert start < end, 'start >= end ({:d}, {:d})'.format(start, end)
                output = self.forward_points(field, embedder, 
                                             points[start: end, ...], 
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
    def forward_points(self, field, embedder, points, directions=None,):
        shape = tuple(points.shape[:-1])  # [B, n_points]
        # [B, 3] -> [B, n_points, 3]
        directions = directions[..., None, :].expand_as(points)
        
        points = points.reshape((-1, 3))
        directions = directions.reshape((-1, 3))

        xyz_embeds = embedder(points)
        if self.dir_embedder is None:
            dir_embeds = None
        else:
            assert self.dir_embedder is not None
            dir_embeds = self.dir_embedder(directions)
        coarse_alphas, coarse_colors = field(xyz_embeds, dir_embeds)
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

