from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmcv.runner import auto_fp16, force_fp32
from vcnerf.core import im2mse, mse2psnr, raw2outputs, SamplePDF
from ..builder import RENDERER, build_embedder, build_field
from .nerf import NeRF


@RENDERER.register_module()
class DynamicNeRF(NeRF):
    def __init__(
        self,
        xyz_embedder,
        t_embedder,
        coarse_can_field,
        coarse_def_field,
        render_params,
        dir_embedder=None,
        fine_can_field=None,
        fine_def_field=None,
    ):
        super().__init__(xyz_embedder, coarse_can_field, render_params, dir_embedder, fine_can_field)
        
        self.t_embedder = t_embedder
        self.coarse_def_field = build_field(coarse_def_field)
        self.fine_def_field = build_field(fine_def_field)

        # sanity checks
        assert self.xyz_embedder.out_dims == coarse_def_field.xyz_emb_dims
        assert self.t_embedder.out_dims == coarse_def_field.t_emb_dims
        if fine_def_field is not None:
            assert self.xyz_embedder.out_dims == fine_def_field.xyz_emb_dims
            assert self.t_embedder.out_dims == fine_def_field.t_emb_dims

    @auto_fp16(apply_to=('points',))
    def forward_points(self, 
                       points, 
                       directions=None, 
                       run_coarse=True, 
                       run_fine=True,
                       **kwarGs):
        shape = tuple(points.shape[:-1])  # [B, n_points]
        # [B, 3] -> [B, n_points, 3]
        directions = directions[..., None, :].expand_as(points)
        
        points = points.reshape((-1, 3))
        directions = directions.reshape((-1, 3))

        if not run_coarse and not run_fine:
            raise ValueError('One or both run_coarse and run_fine should be True')

        t = kwarGs['t']
        t_embeds = self.t_embedder(t)
        xyz_embeds = self.xyz_embedder(points)
        if self.dir_embedder is None:
            dir_embeds = None
        else:
            assert self.dir_embedder is not None
            dir_embeds = self.dir_embedder(directions)

        if run_coarse:
            coarse_dxyz = self.coarse_field(xyz_embeds, t_embeds)
            can_points = points + coarse_dxyz
            can_xyz_embeds = self.xyz_embedder(can_points)
            coarse_alphas, coarse_colors = self.coarse_field(can_xyz_embeds, dir_embeds)
        else:
            coarse_alphas, coarse_colors = None, None
            
        if run_fine and self.fine_field is not None:
            fine_dxyz = self.fine_field(xyz_embeds, t_embeds)
            can_points = points + fine_dxyz
            can_xyz_embeds = self.xyz_embedder(can_points)
            fine_alphas, fine_colors = self.fine_field(can_xyz_embeds, dir_embeds)
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
        t = rays.pop('t')
        render_params['t'] = t
        outputs = self.forward_render(**rays, **render_params)

        im_loss = im2mse(outputs['coarse']['color_map'], rays['rays_color'])
        outputs['coarse_loss'] = im_loss

        if outputs['fine'] is not None:
            im_loss_fine = im2mse(outputs['fine']['color_map'], rays['rays_color'])
            outputs['fine_loss'] = im_loss_fine

        return outputs

