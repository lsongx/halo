from collections import OrderedDict
import enum
from numpy.core.overrides import set_module
from numpy.fft import fftshift
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision.models import vgg16

from mmcv.runner import auto_fp16, force_fp32
from vcnerf.core import im2mse, mse2psnr, raw2outputs, SamplePDF
from ..builder import RENDERER, build_embedder, build_field


@RENDERER.register_module()
class FastNeLF(nn.Module):
    def __init__(self, 
                 st_embedder, 
                 uv_embedder, 
                 st_basis_field,
                 uv_st_field, 
                 render_params,):
        super().__init__()
        self.st_embedder = build_embedder(st_embedder)
        self.uv_embedder = build_embedder(uv_embedder)
        self.st_basis_field = build_field(st_basis_field)
        self.uv_st_field = build_field(uv_st_field)

        self.render_params = render_params
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

        im_loss = im2mse(outputs['color_map'], rays['rays_color'])
        outputs['rec_loss'] = im_loss

        return outputs

    def _parse_outputs(self, outputs):
        loss, log_vars = self._parse_losses(outputs)
        log_vars['psnr'] = mse2psnr(outputs['rec_loss']).item()
        outputs.update(dict(loss=loss, log_vars=log_vars))
        outputs['num_samples'] = 1
        return outputs

    @auto_fp16()
    def forward_render(self,
                       uv, st, 
                       aug_uv, aug_st, 
                       rays_color, h=400, w=400, batch_ray_forward=False):
        if isinstance(h, torch.Tensor):
            h = h[0]
        self.h, self.w = int(h), int(w)
        b, n, _ = uv.shape

        st_embeds = self.st_embedder(st.reshape([-1,2]))
        uv_embeds = self.uv_embedder(uv.reshape([-1,2]))
        if self.training or not batch_ray_forward:
            basis = self.st_basis_field(st_embeds) # [b*n,num_basis,3]
            weights = self.uv_st_field(uv_embeds, st_embeds) # [b*n,num_basis]
            rgb = (basis*weights[:,:,None]).sum(1)
        else:
            rgb = self.batch_ray_forward(uv_embeds, st_embeds, batch_ray_forward)
        outputs = {}
        outputs['color_map'] = rgb.reshape([b, n, 3])

        return outputs

    def batch_ray_forward(self, uv_embeds, st_embeds, batch_ray_forward):
        o = []
        i = 0
        while i < uv_embeds.shape[0]:
            end = min(uv_embeds.shape[0], i+batch_ray_forward)
            t_st_embeds = st_embeds[i:end,...]
            t_uv_embeds = uv_embeds[i:end,...]
            basis = self.st_basis_field(t_st_embeds) # [b*n,num_basis,3]
            weights = self.uv_st_field(t_uv_embeds, t_st_embeds) # [b*n,num_basis]
            rgb = (basis*weights[:,:,None]).sum(1)
            o.append(rgb)
            i += batch_ray_forward
        return torch.cat(o, dim=0)

    def train_step(self, data, optimizer, **kwargs):
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        kwargs['render_params'] = {'batch_ray_forward': 1024}
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

