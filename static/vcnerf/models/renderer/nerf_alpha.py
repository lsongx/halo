from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmcv.runner import auto_fp16, force_fp32
from vcnerf.core import im2mse, mse2psnr, SamplePDF
from ..builder import RENDERER, build_embedder, build_field


def raw2outputs(alphas, colors, z_vals, rays_dir, alpha_noise_std, background):
    # [B, n_points, 1] -> [B, n_points]
    alphas = alphas.squeeze(-1)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    t_dists_mean = dists.mean()
    dists = torch.cat([
        dists, 
        2e10 * torch.ones(dists[..., :1].shape, dtype=torch.float32, device=dists.device)
        # dists.mean() * torch.ones(dists[..., :1].shape, dtype=torch.float32, device=dists.device)
    ], dim=-1)  # [B, n_samples]
    dists = dists * torch.norm(rays_dir[..., None, :], dim=-1)

    alphas = torch.nn.Softplus()(alphas)
    # alphas = 1-torch.exp(-alphas*1e-3) # seems work
    # alphas = 1-torch.exp(-alphas*t_dists_mean) # work
    # alphas = 1-torch.exp(-alphas*dists) # original
    alphas = 1-torch.exp(-alphas) # not work

    cp = torch.cumprod(1-alphas, 1)
    cp = torch.roll(cp, 1, 1)
    cp[:,0] = 1.0
    weights = cp*alphas

    # Computed weighted color of each sample y
    color_map = torch.sum(weights[..., None] * colors, dim=-2)  # [B, 3]

    # Estimated depth map is expected distance.
    depth_map = torch.sum(weights * z_vals, dim=-1)  # [B]

    # Disparity map is inverse depth.
    disp_map = 1 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, dim=-1))

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)

    # To composite onto a white backGround, use the accumulated alpha map
    if background == 'white':
        color_map = color_map + (1 - acc_map[..., None]) 

    outputs = {
        'alphas': alphas, 
        'weights': weights, 
        'color_map': color_map, 
        'depth_map': depth_map, 
        'disp_map': disp_map, 
        'acc_map': acc_map
    }
    return outputs



@RENDERER.register_module()
class NeRFAlpha(nn.Module):
    def __init__(self, 
                 xyz_embedder, 
                 coarse_field, 
                 render_params,
                 dir_embedder=None, 
                 fine_field=None,):
        super().__init__()
        self.xyz_embedder = build_embedder(xyz_embedder)
        self.coarse_field = build_field(coarse_field)
        self.dir_embedder = build_embedder(dir_embedder)
        self.fine_field = build_field(fine_field)
        self.render_params = render_params
        self.sample_pdf = SamplePDF()
        self.fp16_enabled = False
        self.iter = 0

        # sanity checks
        assert self.xyz_embedder.out_dims == coarse_field.xyz_emb_dims
        if fine_field is not None:
            assert self.xyz_embedder.out_dims == fine_field.xyz_emb_dims

        if self.dir_embedder is None:
            assert not self.coarse_field.use_dirs
            assert self.fine_field is None or (not self.fine_field.use_dirs)

            assert self.dir_embedder.out_dims == coarse_field.dir_emb_dims
            if fine_field is not None:
                assert self.dir_embedder.out_dims == fine_field.dir_emb_dims

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

        im_loss = im2mse(outputs['coarse']['color_map'], rays['rays_color'])
        outputs['coarse_loss'] = im_loss

        if outputs['fine'] is not None:
            im_loss_fine = im2mse(outputs['fine']['color_map'], rays['rays_color'])
            outputs['fine_loss'] = im_loss_fine

        return outputs

    def _parse_outputs(self, outputs):
        loss, log_vars = self._parse_losses(outputs)
        if outputs['fine'] is not None:
            log_vars['coarse_psnr'] = mse2psnr(outputs['coarse_loss']).item()
            log_vars['psnr'] = mse2psnr(outputs['fine_loss']).item()
        else:
            log_vars['psnr'] = mse2psnr(outputs['coarse_loss']).item()
        outputs.update(dict(loss=loss, log_vars=log_vars))
        outputs['num_samples'] = 1
        return outputs

    @auto_fp16()
    def forward_render(self,
                       rays_ori, rays_dir, rays_color, ndc_rays_ori, ndc_rays_dir, # loader output
                       n_samples, n_importance, perturb, alpha_noise_std, inv_depth, # render param
                       use_dirs, max_rays_num, ndc=False, near=0.0, far=1.0, background=False):

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
        
        # TODO: double check
        if use_dirs:
            directions = F.normalize(rays_dir, p=2, dim=-1)
        else:
            directions = None
        
        # import pdb; pdb.set_trace()
        if ndc:
            # points in space to evaluate model at
            points = ndc_rays_ori[..., None, :] + ndc_rays_dir[..., None, :] * \
                z_vals[..., :, None]  # [B, n_samples, 3] or [B, H, W, n_samples, 3]
        else:
            # points in space to evaluate model at
            points = rays_ori[..., None, :] + rays_dir[..., None, :] * \
                z_vals[..., :, None]  # [B, n_samples, 3] or [B, H, W, n_samples, 3]

        # if use_dirs:
        #     directions = F.normalize(rays_dir, p=2, dim=-1)
        #     # points in space to evaluate model at
        #     points = ndc_rays_ori[..., None, :] + ndc_rays_dir[..., None, :] * \
        #         z_vals[..., :, None]  # [B, n_samples, 3] or [B, H, W, n_samples, 3]
        # else:
        #     directions = None
        #     # points in space to evaluate model at
        #     points = rays_ori[..., None, :] + rays_dir[..., None, :] * \
        #         z_vals[..., :, None]  # [B, n_samples, 3] or [B, H, W, n_samples, 3]

        # Evaluates the model at the points
        # coarse_alphas, coarse_colors = run_model(model, points, directions, run_coarse=True, run_fine=False)[:2]
        coarse_alphas, coarse_colors = self.forward_batchified(points, 
                                                               directions, 
                                                               run_coarse=True, 
                                                               run_fine=False, 
                                                               max_rays_num=max_rays_num)[:2]
        coarse_outputs = raw2outputs(coarse_alphas, 
                                     coarse_colors, 
                                     z_vals, 
                                     ndc_rays_dir if ndc else rays_dir,
                                     alpha_noise_std,
                                     background)

        if n_importance > 0:
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

            if ndc:
                # points in space to evaluate model at
                points = ndc_rays_ori[..., None, :] + ndc_rays_dir[..., None, :] * \
                    z_vals_fine[..., :, None]  # [B, n_importance, 3]
            else:
                # points in space to evaluate model at
                points = rays_ori[..., None, :] + rays_dir[..., None, :] * \
                    z_vals_fine[..., :, None]  # [B, n_importance, 3]

            # if use_dirs:
            #     directions = F.normalize(rays_dir, p=2, dim=-1)
            #     # points in space to evaluate model at
            #     points = ndc_rays_ori[..., None, :] + ndc_rays_dir[..., None, :] * \
            #         z_vals_fine[..., :, None]  # [B, n_importance, 3]
            # else:
            #     directions = None
            #     # points in space to evaluate model at
            #     points = rays_ori[..., None, :] + rays_dir[..., None, :] * \
            #         z_vals_fine[..., :, None]  # [B, n_importance, 3]
            
            # Evaluates the model at the points
            # fine_alphas0, fine_colors0 = fine_alphas, fine_colors
            max_rays_num = int(max_rays_num * n_samples / (n_samples + n_importance))
            fine_alphas, fine_colors = self.forward_batchified(points, 
                                                               directions, 
                                                               run_coarse=False, 
                                                               run_fine=True,
                                                               max_rays_num=max_rays_num)[2:]
            # fine_alphas = torch.cat([fine_alphas0, fine_alphas], dim=-2)  # [B, n_samples + n_importance, 1]
            # fine_colors = torch.cat([fine_colors0, fine_colors], dim=-2)  # [B, n_samples + n_importance, 3]
            fine_outputs = raw2outputs(fine_alphas, 
                                       fine_colors, 
                                       z_vals_fine, 
                                       ndc_rays_dir if ndc else rays_dir,
                                       alpha_noise_std,
                                       background)
        else:
            fine_outputs = None
        return {'fine': fine_outputs, 'coarse': coarse_outputs, 'color_map': fine_outputs['color_map']}

    @auto_fp16()
    def forward_batchified(self, 
                           points, 
                           directions, 
                           run_coarse, 
                           run_fine, 
                           max_rays_num,):
        assert points.shape[0] == directions.shape[0], (
            f'points: {points.shape}, directions: {directions.shape}')
        nb_rays = points.shape[0]
        if nb_rays <= max_rays_num:
            return self.forward_points(points, directions, run_coarse, run_fine)
        else:
            outputs = []
            start = 0
            end = max_rays_num
            while start < nb_rays:
                assert start < end, 'start >= end ({:d}, {:d})'.format(start, end)
                output = self.forward_points(points[start: end, ...], 
                                             directions[start: end, ...], 
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
                       directions=None, 
                       run_coarse=True, 
                       run_fine=True):
        shape = tuple(points.shape[:-1])  # [B, n_points]
        # [B, 3] -> [B, n_points, 3]
        directions = directions[..., None, :].expand_as(points)
        
        points = points.reshape((-1, 3))
        directions = directions.reshape((-1, 3))

        if not run_coarse and not run_fine:
            raise ValueError('One or both run_coarse and run_fine should be True')

        xyz_embeds = self.xyz_embedder(points)
        if self.dir_embedder is None:
            dir_embeds = None
        else:
            assert self.dir_embedder is not None
            dir_embeds = self.dir_embedder(directions)

        if run_coarse:
            coarse_alphas, coarse_colors = self.coarse_field(xyz_embeds, dir_embeds)
        else:
            coarse_alphas, coarse_colors = None, None
            
        if run_fine and self.fine_field is not None:
            fine_alphas, fine_colors = self.fine_field(xyz_embeds, dir_embeds)
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

    def train_step(self, data, optimizer, **kwargs):
        for k, v in data.items():
            if v.shape[0] == 1:
                data[k] = v[0] # batch size = 1
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        return self.train_step(data, optimizer, **kwargs)


# if __name__ == '__main__':
#     from modules.field import Field
#     from modules.embedder import Embedder

#     coarse_field = Field()
#     fine_field = Field()
#     xyz_embedder = Embedder(3, 10)
#     dir_embedder = Embedder(3, 4)
#     model = NeRF(xyz_embedder, coarse_field, dir_embedder, fine_field)

#     points = torch.rand((2, 3))
#     directions = torch.rand((2, 3))
#     coarse_alphas, coarse_colors, fine_alphas, fine_colors = model(points, directions)
#     print('coarse alphas:', coarse_alphas.shape)
#     print('fine alphas:', fine_alphas.shape)
#     print('coarse colors:', coarse_colors.shape)
#     print('fine colors:', fine_colors.shape)