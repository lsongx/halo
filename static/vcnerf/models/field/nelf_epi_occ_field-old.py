import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16, force_fp32
from ..builder import FIELD


@FIELD.register_module()
class NeLFEPIOccField(nn.Module):
    def __init__(self, 
                 nb_layers=8, 
                 hid_dims=256, 
                 emb_dims=84, 
                 epi_emb_dims=84, 
                 color_dims=32,
                 positive_color_code=True,
                 use_sin=False):
        super().__init__()
        self.nb_layers = nb_layers
        self.hid_dims = hid_dims
        self.emb_dims = emb_dims
        self.epi_emb_dims = epi_emb_dims
        self.color_dims = color_dims
        self.skips = [nb_layers // 2]
        self.positive_color_code = positive_color_code

        self.layers = nn.ModuleDict()
        self.layers.add_module('fc0', nn.Linear(emb_dims*2 + epi_emb_dims, hid_dims))
        for i in range(1, nb_layers):
            if i in self.skips:
                self.layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims + emb_dims*2 + epi_emb_dims, hid_dims)
                )
            else:
                self.layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims, hid_dims)
                )

        self.alpha_out = nn.Linear(hid_dims, 1)
        self.softplus = nn.Softplus()
        self.color_code_out = nn.Linear(hid_dims, color_dims)

        self.fp16_enabled = False
        if use_sin:
            self.activation = torch.sin
        else:
            self.activation = nn.ReLU()

    @auto_fp16()
    def forward(self, uv_embeds, st_embeds, epi_embeds, distance):
        embeds = [uv_embeds, st_embeds]
        all_embeds = torch.cat(embeds+[epi_embeds], dim=-1)
        # x = torch.cat(embeds, dim=-1)
        x = all_embeds
        for i in range(self.nb_layers):
            key = 'fc{}'.format(i)
            layer = self.layers[key]
            if i in self.skips:
                x = torch.cat([x, all_embeds], dim=-1)
            x = layer(x)
            x = self.activation(x)

        alpha = 1-torch.exp(-self.softplus(self.alpha_out(x)-1)*distance)
        color_code = self.color_code_out(x)
        if self.positive_color_code:
            color_code = color_code.sigmoid()
        return alpha, color_code
