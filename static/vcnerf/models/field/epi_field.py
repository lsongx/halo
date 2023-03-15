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
class EPIField(nn.Module):
    def __init__(self, 
                 nb_layers=8, 
                 hid_dims=256, 
                 emb_dims=84, 
                 epi_emb_dims=84, 
                 color_uv_dims=12,
                 use_sin=False):
        super().__init__()
        self.nb_layers = nb_layers
        self.hid_dims = hid_dims
        self.emb_dims = emb_dims
        self.epi_emb_dims = epi_emb_dims
        self.color_uv_dims = color_uv_dims
        self.skips = [nb_layers // 2]

        self.layers = nn.ModuleDict()
        self.layers.add_module('fc0', nn.Linear(emb_dims+epi_emb_dims, hid_dims))
        for i in range(1, nb_layers):
            if i in self.skips:
                self.layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims+emb_dims+epi_emb_dims, hid_dims)
                )
            else:
                self.layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims, hid_dims)
                )

        self.layers['alpha_out'] = nn.Linear(hid_dims, 1)
        self.softplus = nn.Softplus()
        
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
    def forward(self, uv_embeds, st_embeds, epi_embeds, dist):
        all_embeds = torch.cat([st_embeds, epi_embeds], dim=-1)
        x = all_embeds
        for i in range(self.nb_layers):
            key = 'fc{}'.format(i)
            layer = self.layers[key]
            if i in self.skips:
                x = torch.cat([x, all_embeds], dim=-1)
            x = layer(x)
            x = self.activation(x)

        pred_alpha = self.layers['alpha_out'](x)
        if self.training:
            pred_alpha = pred_alpha+torch.randn_like(pred_alpha)
        alpha = 1-torch.exp(-self.activation(pred_alpha)*dist)
        x = torch.cat([x, uv_embeds[...,:self.color_uv_dims]], dim=-1)
        x = self.layers['color_fc'](x)
        x = self.activation(x)
        color = self.layers['color_out'](x)
        color = torch.sigmoid(color)*(1+2*0.001)-0.001
        return alpha, color
