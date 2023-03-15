import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16, force_fp32
from ..builder import FIELD
from .base_field import BaseField


@FIELD.register_module()
class DeformationField(nn.Module):
    def __init__(
        self,
        nb_layers=8,
        hid_dims=256,
        xyz_emb_dims=63,
        t_emb_dims=9
    ):
        super().__init__()
        self.nb_layers = nb_layers
        self.hid_dims = hid_dims
        self.xyz_emb_dims = xyz_emb_dims
        self.t_emb_dims = t_emb_dims
        self.skips = [nb_layers // 2]

        input_dims = xyz_emb_dims + t_emb_dims

        self.layers = nn.ModuleDict()
        self.layers.add_module('fc0', nn.Linear(input_dims, hid_dims))
        for i in range(1, nb_layers):
            if i in self.skips:
                self.layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims + input_dims, hid_dims)
                )
            else:
                self.layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims, hid_dims)
                )

        self.delta_xyz_out = nn.Linear(hid_dims, 3)
        self.fp16_enabled = False
    
    @auto_fp16()
    def forward(self, xyz_embeds, t_embeds):
        input_embeds = torch.cat([xyz_embeds, t_embeds], dim=1)
        x = input_embeds.clone()
        for i in range(self.nb_layers):
            key = 'fc{}'.format(i)
            layer = self.layers[key]
            if i in self.skips:
                x = torch.cat([x, input_embeds], dim=1)
            x = layer(x)
            x = F.relu(x, inplace=True)
        
        delta_xyzs = self.delta_xyz_out(x)
        return delta_xyzs
