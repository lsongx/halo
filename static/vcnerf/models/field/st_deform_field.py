import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16, force_fp32
from ..builder import FIELD


@FIELD.register_module()
class STDeformField(nn.Module):
    def __init__(self, 
                 xyz_embedder=None,
                 t_embedder=None,
                 nb_layers=8, 
                 hid_dims=256, 
                 xyz_emb_dims=63, 
                 dir_emb_dims=27, 
                 t_emb_dims=9, 
                 use_dirs=True):
        super().__init__()
        self.xyz_embedder = xyz_embedder
        self.t_embedder = t_embedder
        self.nb_layers = nb_layers
        self.hid_dims = hid_dims
        self.xyz_emb_dims = xyz_emb_dims
        self.dir_emb_dims = dir_emb_dims
        self.t_emb_dims = t_emb_dims
        self.use_dirs = use_dirs
        self.skips = [nb_layers // 2]

        in_emb_dims = self.xyz_emb_dims+self.t_emb_dims
        self.deform_layers = nn.ModuleDict()
        self.deform_layers.add_module('fc0', nn.Linear(in_emb_dims, hid_dims))
        for i in range(1, nb_layers):
            if i in self.skips:
                self.deform_layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims + in_emb_dims, hid_dims)
                )
            else:
                self.deform_layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims, hid_dims)
                )
        self.deform_out = nn.Linear(hid_dims, 3)

        in_emb_dims = self.xyz_emb_dims
        self.xyz_layers = nn.ModuleDict()
        self.xyz_layers.add_module('fc0', nn.Linear(in_emb_dims, hid_dims))
        for i in range(1, nb_layers):
            if i in self.skips:
                self.xyz_layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims + in_emb_dims, hid_dims)
                )
            else:
                self.xyz_layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims, hid_dims)
                )

        self.alpha_out = nn.Linear(hid_dims, 1)
        if use_dirs:
            self.feat_out = nn.Linear(hid_dims, hid_dims)
            self.color_fc = nn.Linear(hid_dims + dir_emb_dims, hid_dims // 2)
            self.color_out = nn.Linear(hid_dims // 2, 3)
        else:
            self.color_out = nn.Linear(hid_dims, 3)

        self.basis_param = nn.Parameter(torch.rand(3,1,1), requires_grad=True)

        self.no_deform = False
        # self.no_deform = True
        self.fp16_enabled = False

    @auto_fp16()
    def forward(self, points, timestamp, dir_embeds=None):
        # import pdb;pdb.set_trace()
        xyz_embeds = self.xyz_embedder(points)
        t_embeds = self.t_embedder(timestamp)
        if not self.no_deform:
            x = torch.cat([xyz_embeds, t_embeds], dim=1)
            for i in range(self.nb_layers):
                key = 'fc{}'.format(i)
                layer = self.deform_layers[key]
                if i in self.skips:
                    x = torch.cat([x, xyz_embeds, t_embeds], dim=1)
                x = layer(x)
                x = torch.sin(x)
                # x = F.relu(x)
            deformation = self.deform_out(x)
            deformation = 2*(torch.sigmoid(deformation)-0.5)
        else:
            deformation = 0

        new_xyz = points+deformation
        new_xyz_embeds = self.xyz_embedder(new_xyz)
        x = new_xyz_embeds
        for i in range(self.nb_layers):
            key = 'fc{}'.format(i)
            layer = self.xyz_layers[key]
            if i in self.skips:
                x = torch.cat([x, new_xyz_embeds], dim=1)
            x = layer(x)
            x = torch.sin(x)
            # x = F.relu(x)
        alphas_mlp_deform = self.alpha_out(x)
        alphas = self.alpha_out(x)

        has_item_alphas = (alphas_mlp_deform>5)[:,0].detach()
        basis = torch.stack([timestamp[has_item_alphas], 
                             timestamp[has_item_alphas]**2, 
                             timestamp[has_item_alphas]**3], dim=0)
        # algebra_deform = torch.zeros_like(deformation, requires_grad=True)
        # algebra_deform[has_item_alphas, -1] = (self.basis_param*basis).sum(0)[:,0]
        # algebra_deform = (self.basis_param*basis).sum(0)[:,0]
        algebra_deform = (self.basis_param[0]*(timestamp[has_item_alphas]**2)).view([-1])
        algebra_new_xyz = points
        algebra_new_xyz[has_item_alphas, -1] += algebra_deform

        algebra_new_xyz_embeds = self.xyz_embedder(algebra_new_xyz)
        x = algebra_new_xyz_embeds
        for i in range(self.nb_layers):
            key = 'fc{}'.format(i)
            layer = self.xyz_layers[key]
            if i in self.skips:
                x = torch.cat([x, algebra_new_xyz_embeds], dim=1)
            x = layer(x)
            # x = torch.sin(x)
            x = F.relu(x)
        alphas = self.alpha_out(x)

        if self.use_dirs:
            if dir_embeds is None:
                raise ValueError()
            
            feats = self.feat_out(x)
            # NOTE: there is no relu here in the official implementation
            x = torch.cat([feats, dir_embeds], dim=1)
            x = self.color_fc(x)
            x = F.relu(x, inplace=True)
            colors = self.color_out(x)
        else:
            colors = self.color_out(x)
        colors = torch.sigmoid(colors)

        # tmask = (points[has_item_alphas][:,0].abs()<0.15) & (points[has_item_alphas][:,1].abs()<0.15) & (points[has_item_alphas][:,2].abs()>0.24) & (points[has_item_alphas][:,2].abs()<0.30)
        # tmask = (new_xyz[has_item_alphas][:,0].abs()<0.15) & (new_xyz[has_item_alphas][:,1].abs()<0.15) & (new_xyz[has_item_alphas][:,2].abs()>0.24) & (new_xyz[has_item_alphas][:,2].abs()<0.30)
        # tmask = (new_xyz[:,0].abs()<0.15) & (new_xyz[:,1].abs()<0.15) & (new_xyz[:,2].abs()>0.24) & (new_xyz[:,2].abs()<0.30)
        # tmask = colors[:,2]*255<180; tmask = colors[:,2]*255>100; tmask=tmask&(alphas[:,0]>30)
        # if tmask.sum()>0:
        #     print(new_xyz[tmask][:,2])
        #     print(deformation[tmask][:,2])
        #     print(deformation[tmask])
        #     import ipdb;ipdb.set_trace()

        # # torch.save(tmask,'data/out/tmask')
        # # torch.save(has_item_alphas,'data/out/has_item_alphas')
        # # tmask=torch.load('data/out/tmask')
        # # has_item_alphas=torch.load('data/out/has_item_alphas')
        # points[has_item_alphas][tmask]
        # deformation[has_item_alphas][tmask]
        # points[tmask]
        # colors[has_item_alphas][tmask]
        # colors[tmask]*255
        # new_xyz[tmask]
        # alphas[tmask]

        return alphas, colors

