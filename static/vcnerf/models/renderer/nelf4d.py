from collections import OrderedDict
import enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision.models import vgg16

from mmcv.runner import auto_fp16, force_fp32
from vcnerf.core import im2mse, mse2psnr, raw2outputs, SamplePDF
from ..builder import RENDERER, build_embedder, build_field


class Nerf4D(torch.nn.Module):
    def __init__(self, input_dims=4, output_dims=3, scale=16, samples=256, width=512, hidden_layers=6):
        super(Nerf4D, self).__init__()
        
        # self.model = nn.ModuleList()
        # # head
        # self.model.append(nn.Linear(2 * samples, width))
        # self.model.append(nn.GroupNorm(width//16, width))
        # self.model.append(nn.GELU())
        # # hidden
        # for i in range(hidden_layers):
        #     self.model.append(nn.Linear(width, width))
        #     self.model.append(nn.GroupNorm(width//16, width))
        #     self.model.append(nn.GELU())
        # # tail
        # self.model.append(nn.Linear(width, 128))
        # self.model.append(nn.GroupNorm(128//16, 128))
        # self.model.append(nn.GELU())
        # self.model.append(nn.Linear(128, output_dims))
        # self.model.append(nn.Sigmoid())

        # embedding coefficients
        self.b = torch.nn.Parameter(torch.normal(mean=0.0, std=scale, size=(samples, 4)), requires_grad=False)


        nb_layers = 8
        hid_dims = 256
        emb_dims = 256
        self.skips = [nb_layers // 2]

        # self.freqs = torch.normal(mean=0.0, std=16, size=(32,))
        # self.funcs = [torch.sin, torch.cos]
        # emb_dims = 260-4

        self.layers = nn.ModuleDict()
        self.layers.add_module('fc0', nn.Linear(emb_dims, hid_dims))
        for i in range(1, nb_layers):
            if i in self.skips:
                self.layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims + emb_dims, hid_dims)
                )
            else:
                self.layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims, hid_dims)
                )

        self.color_out = nn.Linear(hid_dims, 3)
        self.fp16_enabled = False
        self.activation = nn.ReLU()

    def forward(self, x):

        x = (2.0 * np.pi * x) @ self.b.T
        o1 = torch.sin(x)
        o2 = torch.cos(x)
        z = torch.cat([o1, o2], dim=-1)
        embed = z

        # embeds = [] 
        # for freq in self.freqs:
        #     freq = freq.unsqueeze(0).to(x.device).to(x.dtype)
        #     for func in self.funcs:
        #         embeds.append(func(x * freq))
        # embed = torch.cat(embeds, dim=1)

        x = embed
        for i in range(8):
            key = 'fc{}'.format(i)
            layer = self.layers[key]
            if i in self.skips:
                x = torch.cat([x, embed], dim=1)
            x = layer(x)
            x = self.activation(x)
        colors = self.color_out(x)
        colors = torch.sigmoid(colors)
        return colors

        # end of embedding

        for idx, layer in enumerate(self.model):
            z = layer(z)
            if idx in [8]:
                z += embed
        return z * 1.2 - 0.1


@RENDERER.register_module()
class NeLF4D(nn.Module):
    def __init__(self, 
                 embedder, 
                 field, 
                 render_params,):
        super().__init__()
        self.field = Nerf4D(samples=128, width=256, hidden_layers=6)

        self.render_params = render_params
        self.fp16_enabled = False
        self.iter = 0

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

        # with torch.no_grad():
        #     image = rays['rays_color'].reshape([-1,self.h,self.w,3]).permute([0,3,1,2])
        #     image_fft = torch.fft.rfft(image, 2, norm="forward")
        # aug_image = outputs['aug_color_map'].reshape([-1,self.h,self.w,3]).permute([0,3,1,2])
        # aug_image_fft = torch.fft.rfft(aug_image, 2, norm="forward")
        # outputs['fourier_loss'] = ((image_fft.abs()-aug_image_fft.abs())**2).mean() * 0

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
        uvst = torch.cat([uv, st], dim=-1).reshape([-1, 4]) # [b*n,4]
        embeds = uvst
        if self.training or not batch_ray_forward:
            rgb = self.field(embeds)
        else:
            rgb = self.batch_ray_forward(embeds, batch_ray_forward)
        outputs = {}
        outputs['color_map'] = rgb.reshape([b, n, 3])

        # aug_uvst = torch.cat([aug_uv, aug_st], dim=-1).reshape([-1, 4]) # [b*n,4]
        # aug_embeds = aug_uvst
        # if self.training or not batch_ray_forward:
        #     aug_rgb = self.field(aug_embeds)
        # else:
        #     aug_rgb = self.batch_ray_forward(aug_embeds, batch_ray_forward)
        # outputs['aug_color_map'] = aug_rgb.reshape([b, n, 3])
        outputs['aug_color_map'] = None

        return outputs

    def batch_ray_forward(self, embeds, batch_ray_forward):
        o = []
        i = 0
        while i < embeds.shape[0]:
            end = min(embeds.shape[0], i+batch_ray_forward)
            o.append(self.field(embeds[i:end,...]))
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

