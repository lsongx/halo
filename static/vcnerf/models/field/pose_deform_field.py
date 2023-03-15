import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16, force_fp32
from ..builder import FIELD


@FIELD.register_module()
class PoseDeformField(nn.Module):
    def __init__(self, 
                 nb_layers=8, 
                 hid_dims=256, 
                 input_dims=63, 
                 num_joints=65,
                 use_sin=False):
        super().__init__()
        self.nb_layers = nb_layers
        self.hid_dims = hid_dims
        self.input_dims = input_dims
        self.skips = [nb_layers // 2]

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

        self.weight_out = nn.Linear(hid_dims, num_joints)
        self.fp16_enabled = False
        if use_sin:
            self.activation = torch.sin
        else:
            self.activation = nn.ReLU()

    @auto_fp16()
    def forward(self, x):
        x0 = x.clone()
        for i in range(self.nb_layers):
            key = 'fc{}'.format(i)
            layer = self.layers[key]
            if i in self.skips:
                x = torch.cat([x, x0], dim=1)
            x = layer(x)
            x = self.activation(x)

        joints_weight = self.weight_out(x)
        return joints_weight.softmax(dim=1)
