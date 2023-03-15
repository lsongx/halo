import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16, force_fp32
from ..builder import FIELD


# SIREN from https://github.com/vsitzmann/siren
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-math.sqrt(6/num_input)/30, math.sqrt(6/num_input)/30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1/num_input, 1/num_input)


@FIELD.register_module()
class NeLFEPIField(nn.Module):
    def __init__(self, 
                 nb_layers=8, 
                 hid_dims=256, 
                 emb_dims=84, 
                 color_uv_dims=12,
                 use_sin=False):
        super().__init__()
        self.nb_layers = nb_layers
        self.hid_dims = hid_dims
        self.emb_dims = emb_dims
        self.color_uv_dims= color_uv_dims
        self.skips = [nb_layers // 2]

        self.layers = nn.ModuleDict()
        self.layers.add_module('fc0', nn.Linear(emb_dims*2, hid_dims))
        for i in range(1, nb_layers):
            if i in self.skips:
                self.layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims + emb_dims*2, hid_dims)
                )
            else:
                self.layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims, hid_dims)
                )

        self.layers['nelf_epi_fc'] = nn.Linear(hid_dims, hid_dims)
        self.layers['nelf_epi_out'] = nn.Linear(hid_dims, 1)

        self.layers['color_fc'] = nn.Linear(hid_dims+self.color_uv_dims, hid_dims)
        self.layers['color_out'] = nn.Linear(hid_dims, 3)

        self.fp16_enabled = False
        if use_sin:
            self.activation = lambda x: torch.sin(30*x)
            self.layers.apply(sine_init)
            self.layers.fc0.apply(first_layer_sine_init)
        else:
            self.activation = nn.ReLU()


    @auto_fp16()
    def forward(self, embeds, code_embedder=None, near=None, far=None):
        uv_embeds, st_embeds = embeds
        x = torch.cat(embeds, dim=-1)
        all_embeds = x
        for i in range(self.nb_layers):
            key = 'fc{}'.format(i)
            layer = self.layers[key]
            if i in self.skips:
                x = torch.cat([x, all_embeds], dim=-1)
            x = layer(x)
            x = self.activation(x)

        epi_x = self.layers['nelf_epi_fc'](x)
        epi_x = self.activation(epi_x)
        nelf_epi = self.layers['nelf_epi_out'](epi_x).sigmoid()*(1+2*0.001)-0.001
        if near is not None and far is not None:
            nelf_epi = nelf_epi*(far-near) + near
        
        if self.color_uv_dims>0:
            x = torch.cat([x, uv_embeds[...,:self.color_uv_dims]], dim=-1)
        x = self.layers['color_fc'](x)
        x = self.activation(x)
        color = self.layers['color_out'](x)
        color = torch.sigmoid(color)*(1+2*0.001)-0.001
        return nelf_epi, color
