from collections import OrderedDict
import pdb
from mmcv.utils.misc import import_modules_from_strings
from numpy.lib.arraysetops import isin
from numpy.lib.index_tricks import ndenumerate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmcv.runner import auto_fp16, force_fp32
from vcnerf.core import im2mse, mse2psnr, raw2outputs, SamplePDF
from ..builder import RENDERER, build_embedder, build_field
from .nerf import NeRF


@RENDERER.register_module()
class DeformNeRF(nn.Module):
    def __init__(self,
                 xyz_embedder,
                 t_embedder,
                 coarse_field,
                 render_params,
                 dir_embedder=None,
                 fine_field=None,
                 init_iters=500):
        super().__init__()
        self.xyz_embedder = build_embedder(xyz_embedder)
        self.t_embedder = build_embedder(t_embedder)
        self.coarse_field = build_field(coarse_field)
        self.dir_embedder = build_embedder(dir_embedder)
        self.fine_field = build_field(fine_field)
        self.render_params = render_params
        self.sample_pdf = SamplePDF()

        self.coarse_field.xyz_embedder = self.xyz_embedder
        self.coarse_field.t_embedder = self.t_embedder

        self.fp16_enabled = False
        self.iter = 0
        self.init_iters = init_iters

    def _parse_outputs(self, outputs):
        loss, log_vars = self._parse_losses(outputs)
        # if outputs['fine'] is not None:
        #     log_vars['coarse_psnr'] = mse2psnr(outputs['coarse_loss']).item()
        #     log_vars['psnr'] = mse2psnr(outputs['fine_loss']).item()
        # else:
        #     log_vars['psnr'] = mse2psnr(outputs['coarse_loss']).item()
        outputs.update(dict(loss=loss, log_vars=log_vars))
        outputs['num_samples'] = 1
        return outputs

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

    @auto_fp16()
    def forward_render(self,
                       rays_ori, rays_dir, rays_color, timestamp, # loader output
                       n_samples, n_importance, perturb, alpha_noise_std, inv_depth, # render param
                       use_dirs, max_rays_num, near=0.0, far=1.0, background=False, **kwargs):
        timestamp = timestamp[:,None].float().expand(rays_ori.shape[:2]).reshape([-1,1])
        rays_ori, rays_dir = rays_ori.reshape([-1,3]), rays_dir.reshape([-1,3])

        if isinstance(near, torch.Tensor):
            near = near[0].item()
            far = far[0].item()

        # near, far: [B] or [B, H, W]
        near_seq = near * torch.ones(rays_ori.shape[:-1], dtype=torch.float32, device=rays_ori.device)
        far_seq = far * torch.ones(rays_ori.shape[:-1], dtype=torch.float32, device=rays_ori.device)
        t_vals = torch.linspace(0, 1, n_samples, dtype=torch.float32, device=rays_ori.device)
        if not inv_depth:
            # z_vals: [B, n_samples] or [B, H, W, n_samples]
            z_vals = near_seq[..., None] * (1 - t_vals) + far_seq[..., None] * t_vals
        else:
            z_vals = 1/(1 / near_seq[..., None] * (1 - t_vals) + 1 / far_seq[..., None] * t_vals)

        # Perturbs points coordinates
        if perturb:
            # Gets intervals
            mid_points = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mid_points, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mid_points], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, dtype=torch.float32, device=rays_ori.device)
            z_vals = lower + (upper - lower) * t_rand

        # TODO: double check
        if use_dirs:
            directions = F.normalize(rays_dir, p=2, dim=-1)
        else:
            directions = None

        # points in space to evaluate model at
        points = rays_ori[..., None, :] + rays_dir[..., None, :] * \
            z_vals[..., :, None]  # [B, n_samples, 3] or [B, H, W, n_samples, 3]

        coarse_alphas, coarse_colors = self.forward_batchified(points, 
                                                               timestamp,
                                                               directions, 
                                                               run_coarse=True, 
                                                               run_fine=False, 
                                                               max_rays_num=max_rays_num)[:2]
        coarse_outputs = raw2outputs(coarse_alphas, 
                                     coarse_colors, 
                                     z_vals, 
                                     rays_dir,
                                     alpha_noise_std,
                                     background)

        if n_importance > 0 and self.fine_field is not None:
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = self.sample_pdf(
                z_vals_mid, coarse_outputs['weights'][..., 1:-1], n_importance, not perturb)
            z_vals_fine, indices = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
            z_vals_fine = z_vals_fine.detach()

             # TODO: double check
            if use_dirs:
                directions = F.normalize(rays_dir, p=2, dim=-1)
            else:
                directions = None

            points = rays_ori[..., None, :] + rays_dir[..., None, :] * \
                z_vals_fine[..., :, None]  # [B, n_importance, 3]

            max_rays_num = int(max_rays_num * n_samples / (n_samples + n_importance))
            fine_alphas, fine_colors = self.forward_batchified(points, 
                                                               timestamp, 
                                                               directions, 
                                                               run_coarse=False, 
                                                               run_fine=True,
                                                               max_rays_num=max_rays_num)[2:]
            fine_outputs = raw2outputs(fine_alphas, 
                                       fine_colors, 
                                       z_vals_fine, 
                                       rays_dir,
                                       alpha_noise_std,
                                       background)
        else:
            fine_outputs = None
        return {'fine': fine_outputs, 'coarse': coarse_outputs}

    @auto_fp16()
    def forward_batchified(self, 
                           points, 
                           timestamp,
                           directions, 
                           run_coarse, 
                           run_fine, 
                           max_rays_num,):
        nb_rays = points.shape[0]
        if nb_rays <= max_rays_num or self.training:
            return self.forward_points(points, timestamp, directions, run_coarse, run_fine)
        else:
            outputs = []
            start = 0
            end = max_rays_num
            while start < nb_rays:
                if directions is not None:
                    directions_interval = directions[start: end, ...]
                else:
                    directions_interval = directions
                assert start < end, 'start >= end ({:d}, {:d})'.format(start, end)
                output = self.forward_points(points[start: end, ...], 
                                             timestamp[start: end, ...],
                                             directions_interval, 
                                             run_coarse, 
                                             run_fine,)
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
    def forward_points(self, 
                       points, 
                       timestamp,
                       directions=None, 
                       run_coarse=True, 
                       run_fine=True,
                       **kwargs):
        shape = tuple(points.shape[:-1])  # [B, n_points]
        # [B, 3] -> [B, n_points, 3]
        if directions is not None:
            directions = directions[..., None, :].expand_as(points)
        timestamp = timestamp[:,None].expand([*shape, 1])

        points = points.reshape((-1, 3))
        if directions is not None:
            directions = directions.reshape((-1, 3))
        timestamp = timestamp.reshape([-1,1])

        if not run_coarse and not run_fine:
            raise ValueError('One or both run_coarse and run_fine should be True')

        # t_embeds = self.t_embedder(timestamp)
        # xyz_embeds = self.xyz_embedder(points)
        if self.dir_embedder is None:
            dir_embeds = None
        else:
            assert self.dir_embedder is not None
            dir_embeds = self.dir_embedder(directions)

        if run_coarse:
            coarse_alphas, coarse_colors = self.coarse_field(points, timestamp, dir_embeds)
        else:
            coarse_alphas, coarse_colors = None, None
            
        if run_fine and self.fine_field is not None:
            fine_alphas, fine_colors = self.fine_field(points, timestamp, dir_embeds)
        else:
            fine_alphas, fine_colors = None, None

        if coarse_alphas is not None:
            # [B, n_points, 1/3]
            coarse_alphas = coarse_alphas.reshape(shape + (1,))
            coarse_colors = coarse_colors.reshape(shape + (3,))
        if fine_alphas is not None:
            fine_alphas = fine_alphas.reshape(shape + (1,))
            fine_colors = fine_colors.reshape(shape + (3,))

        return coarse_alphas, coarse_colors, fine_alphas, fine_colors

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

        if self.iter<self.init_iters*0.7:
            self.coarse_field.no_deform = True
        else:
            if self.init_iters>0:
                self.coarse_field.no_deform = False
                for n, p in self.coarse_field.named_parameters():
                    if 'deform_layers' not in n:
                        p.requires_grad = False

        outputs = self.forward_render(**rays, **render_params)

        im_loss = (outputs['coarse']['color_map']-rays['rays_color'].view([-1,3]))**2
        outputs['coarse_loss'] = im_loss.mean()
        if self.iter<self.init_iters*0.7:
            background_mask = (rays['rays_color'].sum(2)==0).detach().view(-1)
            acc_loss = ((outputs['coarse']['acc_map'][background_mask])**2).mean()
            outputs['acc_loss'] = acc_loss*0.1

        # if self.iter>self.init_iters/4:
        #     reg_mask = (outputs['coarse']['alphas'].abs()<30).detach()
        #     reg_mask = reg_mask & (outputs['coarse']['alphas']>1).detach()
        #     if reg_mask.sum()>0:
        #         margin_loss30 = (30-outputs['coarse']['alphas'][reg_mask])**2
        #         margin_loss1 = (1-outputs['coarse']['alphas'][reg_mask])**2
        #         outputs['alpha_loss'] = torch.min(margin_loss1, margin_loss30).detach()
        #     else:
        #         outputs['alpha_loss'] = torch.zeros_like(im_loss)

        if outputs['fine'] is not None:
            im_loss_fine = (outputs['fine']['color_map']-rays['rays_color'].view([-1,3]))**2
            outputs['fine_loss'] = im_loss_fine.mean()

        return outputs

    def train_step(self, data, optimizer, **kwargs):
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

