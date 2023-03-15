from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmcv.runner import auto_fp16, force_fp32
from vcnerf.core import im2mse, mse2psnr, raw2outputs, SamplePDF
from ..builder import (RENDERER, build_embedder, build_field, 
                       Registry, build_from_cfg)

SPN = Registry('space_proposal_network')


@SPN.register_module()
class NaiveSPN(nn.Module):
    def __init__(self, in_dims, nb_freqs, include_input=True):
        super().__init__()
        self.in_dims = in_dims
        self.nb_freqs = nb_freqs
        self.include_input = include_input
        self.out_dims = (2 * in_dims * nb_freqs + in_dims) \
            if include_input else (2 * in_dims * nb_freqs)

        self.freqs = 2 ** torch.linspace(0, self.nb_freqs-1, self.nb_freqs)
        self.funcs = [torch.sin, torch.cos]
        self.fp16_enabled = False

        input_channel = len(self.funcs)*len(self.freqs)+1
        self.up_conv = nn.Sequential(
            self.get_interpolate_conv(4, 12*input_channel, 512, 3, 1, 1), # hw[4,4]
            self.get_interpolate_conv(4, 512, 256, 3, 1, 1), # hw[16,16]
            self.get_interpolate_conv(4, 256, 128, 3, 1, 1), # hw[64,64]
            self.get_interpolate_conv(4, 128, 64, 3, 1, 1), # hw[256,256]
        )
        self.conv_after_interpolate = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), 
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), 
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 2, 1), 
        )

    def get_interpolate_conv(self, scale, i, o, k, s, p):
        return nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear'),
            nn.Conv2d(i, o, k, s, p, bias=False),
            nn.InstanceNorm2d(o),
            nn.LeakyReLU(),
        )

    @auto_fp16()
    def forward(self, inputs, h, w):
        inputs = inputs.flatten(1) # [B,12]
        device = inputs.device
        embeds = [inputs] if self.include_input else []
        for freq in self.freqs:
            freq = freq.unsqueeze(0).to(device).to(inputs.dtype)
            for func in self.funcs:
                embeds.append(func(inputs * freq))
        embeds = torch.cat(embeds, dim=1)[..., None, None] # [B,12*freq,1,1]
        feat = self.up_conv(embeds)
        feat = torch.nn.functional.interpolate(feat, size=(h,w))
        return self.conv_after_interpolate(feat)



@RENDERER.register_module()
class FastNeRF(nn.Module):
    def __init__(self, 
                 space_proposal_network,
                 xyz_embedder, 
                 dir_embedder, 
                 field,
                 render_params,):
        super().__init__()
        self.spn = build_from_cfg(space_proposal_network, SPN)
        self.xyz_embedder = build_embedder(xyz_embedder)
        self.dir_embedder = build_embedder(dir_embedder)
        self.field = build_field(field)
        self.render_params = render_params
        self.sample_pdf = SamplePDF()
        self.fp16_enabled = False
        self.iter = 0
        self.self_supervised_depth = True

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
        outputs['im_loss'] = im_loss
        if self.self_supervised_depth:
            depth = outputs['coarse']['depth_map'].detach()
            if depth.shape[0] == 1:
                depth = depth[0]
            outputs['depth_loss'] = F.l1_loss(outputs['coarse']['spn_depth'], depth)

        return outputs

    def _parse_outputs(self, outputs):
        loss, log_vars = self._parse_losses(outputs)
        log_vars['psnr'] = mse2psnr(outputs['im_loss']).item()
        outputs.update(dict(loss=loss, log_vars=log_vars))
        outputs['num_samples'] = 1
        return outputs

    @auto_fp16()
    def forward_render(self,
                       rays_ori, rays_dir, rays_color, ndc_rays_ori, ndc_rays_dir, h, w, pose, select_mask, # loader output 
                       n_samples, perturb, alpha_noise_std, inv_depth, # render param
                       use_dirs, max_rays_num, ndc=False, near=0.0, far=1.0, background=False):

        assert pose.shape[0] == 1
        depth_delta = self.spn(pose, int(h[0].item()), int(w[0].item()))
        depth_selected = depth_delta[0, 0][select_mask[0].bool()]
        delta_selected = depth_delta[0, 1][select_mask[0].bool()]

        # 1. initialization of the network
        # 2. two parts; gradually sparse the fixed points; teacher directly teaching the proposal network

        if not self.self_supervised_depth:
            # use proposal based xyz
            # TODO
            pass
        else:
            # generate xyz grid
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

        # Evaluates the model at the points
        alphas, colors = self.forward_batchified(points, 
                                                 directions, 
                                                 run_coarse=True, 
                                                 run_fine=False, 
                                                 max_rays_num=max_rays_num)[:2]
        outputs = raw2outputs(alphas, 
                              colors, 
                              z_vals, 
                              ndc_rays_dir if ndc else rays_dir,
                              alpha_noise_std,
                              background)

        if self.self_supervised_depth:
            outputs['spn_depth'] = depth_selected

        return {'coarse': outputs, 'fine': None}

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
                       directions=None,):
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

        alphas, colors = self.field(xyz_embeds, dir_embeds)

        if alphas is not None:
            # [B, n_points, 1/3]
            alphas = alphas.reshape(shape + (1,))
            colors = colors.reshape(shape + (3,))

        return alphas, colors

    def train_step(self, data, optimizer, **kwargs):
        ignore_list = ['pose', 'h', 'w', 'select_mask']
        for k, v in data.items():
            if k not in ignore_list and v.shape[0] == 1:
                data[k] = v[0] # batch size = 1
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        return self.train_step(data, optimizer, **kwargs)

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