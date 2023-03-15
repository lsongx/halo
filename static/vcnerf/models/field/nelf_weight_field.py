import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16, force_fp32
from ..builder import FIELD


@FIELD.register_module()
class NeLFWeightField(nn.Module):
    def __init__(self, 
                 nb_layers=6, 
                 hid_dims=256, 
                 st_emb_dims=64, 
                 uv_emb_dims=64, 
                 out_dims=128,
                 use_sin=False):
        super().__init__()
        self.nb_layers = nb_layers
        self.hid_dims = hid_dims
        self.uv_emb_dims = uv_emb_dims
        self.st_emb_dims = st_emb_dims
        self.skips = [nb_layers // 2]

        self.layers = nn.ModuleDict()
        self.layers.add_module('fc0', nn.Linear(uv_emb_dims, hid_dims))
        for i in range(1, nb_layers):
            if i in self.skips:
                self.layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims + uv_emb_dims, hid_dims)
                )
            else:
                self.layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims, hid_dims)
                )

        self.feat_out = nn.Linear(hid_dims, hid_dims)
        self.weight_fc = nn.Linear(hid_dims+st_emb_dims, hid_dims//2)
        self.weight_out = nn.Linear(hid_dims//2, out_dims)
        self.fp16_enabled = False
        if use_sin:
            self.activation = torch.sin
        else:
            self.activation = nn.ReLU()

    @auto_fp16()
    def forward(self, uv_embeds, st_embeds):
        x = uv_embeds
        for i in range(self.nb_layers):
            key = 'fc{}'.format(i)
            layer = self.layers[key]
            if i in self.skips:
                x = torch.cat([x, uv_embeds], dim=1)
            x = layer(x)
            x = self.activation(x)
        
        feat = self.feat_out(x)
        x = torch.cat([feat, st_embeds], dim=1)
        x = self.weight_fc(x)
        x = self.activation(x)
        weight = self.weight_out(x)
        return weight.softmax(dim=1)
