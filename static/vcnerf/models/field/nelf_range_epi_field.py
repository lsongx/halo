import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16, force_fp32
from ..builder import FIELD


@FIELD.register_module()
class NeLFRangeEPIField(nn.Module):
    def __init__(self, 
                 nb_layers=8, 
                 hid_dims=256, 
                 emb_dims=84, 
                 color_dims=32,
                 color_uv_dims=12,
                 use_sin=False):
        super().__init__()
        self.nb_layers = nb_layers
        self.hid_dims = hid_dims
        self.emb_dims = emb_dims
        self.color_dims = color_dims
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

        self.epi_dir_out = nn.Linear(hid_dims, 2)
        self.color_code_out = nn.Linear(hid_dims, color_dims)
        self.range_out = nn.Linear(hid_dims, 2)

        self.color_fc = nn.Linear(color_dims+self.color_uv_dims, hid_dims)
        self.color_out = nn.Linear(hid_dims, 3)
        self.fp16_enabled = False
        if use_sin:
            self.activation = torch.sin
        else:
            self.activation = nn.ReLU()

    @auto_fp16()
    def forward(self, embeds):
        uv_embeds, st_embeds = embeds
        x = torch.cat(embeds, dim=1)
        all_embeds = x
        for i in range(self.nb_layers):
            key = 'fc{}'.format(i)
            layer = self.layers[key]
            if i in self.skips:
                x = torch.cat([x, all_embeds], dim=1)
            x = layer(x)
            x = self.activation(x)
        
        epi_dir = self.epi_dir_out(x).softmax(dim=1)
        color_code = self.color_code_out(x)
        epi_range = self.range_out(x).sigmoid()

        x = torch.cat([color_code, uv_embeds[:,:self.color_uv_dims]], dim=1)
        x = self.color_fc(x)
        x = self.activation(x)
        colors = self.color_out(x)
        colors = torch.sigmoid(colors)
        return epi_dir, epi_range, color_code, colors
