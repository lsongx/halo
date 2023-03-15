import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16, force_fp32
from ..builder import FIELD


class HFSubNet(nn.Module):
    def __init__(self, in_dims, hid_dims=256, layers=4, use_sin=False):
        super().__init__()
        self.layers = nn.ModuleDict()
        self.layers.add_module('fc0', nn.Linear(in_dims, hid_dims))
        for i in range(1, layers):
            self.layers.add_module(f'fc{i}', nn.Linear(hid_dims, hid_dims))
        if use_sin:
            self.activation = torch.sin
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        for k, v in self.layers.items():
            x = v(x)
            x = self.activation(x)
        return x


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
                 color_dims=32,
                 color_uv_dims=12,
                 positive_color_code=False,
                 hf_color_subnet_dim=0,
                 use_sin=False):
        super().__init__()
        self.nb_layers = nb_layers
        self.hid_dims = hid_dims
        self.emb_dims = emb_dims
        self.color_dims = color_dims
        self.color_uv_dims= color_uv_dims
        self.skips = [nb_layers // 2]
        self.positive_color_code = positive_color_code

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

        self.epi_dir_fc = nn.Linear(hid_dims, hid_dims)
        self.epi_dir_out = nn.Linear(hid_dims, 1)
        self.color_code_out = nn.Linear(hid_dims, color_dims)

        self.color_fc = nn.Linear(color_dims+self.color_uv_dims, hid_dims)
        self.fp16_enabled = False
        if use_sin:
            self.activation = lambda x: torch.sin(30*x)
            self.layers.apply(sine_init)
            self.layers.fc0.apply(first_layer_sine_init)
        else:
            self.activation = nn.ReLU()

        if hf_color_subnet_dim > 0:
            self.color_code_hf_subnet = HFSubNet(
                hf_color_subnet_dim, hid_dims, nb_layers//2, use_sin)
            self.color_out = nn.Linear(hid_dims*2, 3)
        else:
            self.color_code_hf_subnet = None
            self.color_out = nn.Linear(hid_dims, 3)

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

        epi_x = self.epi_dir_fc(x)
        epi_x = self.activation(epi_x)
        epi_dir = self.epi_dir_out(epi_x).sigmoid()*(1+2*0.001)-0.001
        if near is not None:
            epi_dir = epi_dir*(far-near) + near
        color_code = self.color_code_out(x)
        if self.positive_color_code:
            color_code = color_code.sigmoid()

        colors = self.forward_with_color_code(color_code, uv_embeds, code_embedder)
        return epi_dir, color_code, colors

    @auto_fp16()
    def forward_with_color_code(self, color_code, uv_embeds, code_embedder=None):
        x = torch.cat([color_code, uv_embeds[...,:self.color_uv_dims]], dim=-1)
        x = self.color_fc(x)
        x = self.activation(x)

        if self.color_code_hf_subnet is not None:
            hf_code = code_embedder(color_code)
            hf_x = self.color_code_hf_subnet(hf_code)
            x = torch.cat([x, hf_x], dim=-1)

        colors = self.color_out(x)
        colors = torch.sigmoid(colors)*(1+2*0.001)-0.001
        return colors

