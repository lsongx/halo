from collections import OrderedDict
import random
import numpy as np
from numpy.core.numeric import outer
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.modules import distance

from torchvision.models import vgg16

from mmcv.runner import auto_fp16, force_fp32
from vcnerf.core import im2mse, mse2psnr, raw2outputs, SamplePDF
from ..builder import RENDERER, build_embedder, build_field

from .nelf_epi_occ import NeLFEPIOcc


def get_intersect_points_on_sphere(point1, point2, r2):
    # return the point nearest to point1
    # http://www.ambrsoft.com/TrigoCalc/Sphere/SpherLineIntersection_.htm
    # a = (x2 − x1)2 + (y2 − y1)2 + (z2 − z1)2
    # b = − 2[(x2 − x1)(xc − x1) + (y2 − y1)(yc − y1) + (z2 − z1)(zc − z1)]
    # c = (xc − x1)2 + (yc − y1)2 + (zc − z1)2 − r2
    # t = (-b±√(b^2-4ac))/2a

    dir = point1-point2
    ori = point1
    
    a = (dir**2).sum(dim=-1)
    b = -2 * (dir * (-ori)).sum(dim=-1)
    c = (ori**2).sum(dim=-1) - r2
    delta = (b**2-4*a*c) > 0
    import pdb;pdb.set_trace()
    if not torch.all(delta):
        raise RuntimeError('not all deltas are positive')

    t1 = (-b+torch.sqrt(b**2-4*a*c)) / (2*a)
    t2 = (-b-torch.sqrt(b**2-4*a*c)) / (2*a)

    intersect_point1 = ori+dir*t1[:,None]
    intersect_point2 = ori+dir*t2[:,None]

    point1_dist = (intersect_point1-ori).abs().sum()
    point2_dist = (intersect_point2-ori).abs().sum()
    if point1_dist > point2_dist:
        point_selected = intersect_point2
    else:
        point_selected = intersect_point1
    # always use the nearest one, convert from points to coord

    # (x,y) as coord
    sphere_coord = point_selected[:,:2]

    return sphere_coord


@RENDERER.register_module()
class NeLFEPIOcc360(NeLFEPIOcc):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_st_by_epi(self, ori_uv, ori_st, new_uv, epi):
        with torch.no_grad():
            # 1. find line from ori uv and ori st
            # --> uv st to 3d points
            st_z = (self.rad0-(ori_st**2).sum(-1,True)).sqrt()
            uv_z = (self.rad1-(ori_uv**2).sum(-1,True)).sqrt()
            st_points = torch.cat([ori_st, st_z], dim=-1)
            uv_points = torch.cat([ori_uv, uv_z], dim=-1)
            # 2. find point on line by epi
            # what if no intersection between the ray and 
            epi_points_xy = (epi-st_z) * ((ori_uv-ori_st) / (uv_z-st_z)) + ori_st
            epi_points = torch.cat([epi_points_xy, epi], dim=-1)
            # 3. adjust epi far (can we?)

            # 4. get intersection on st sphere
            new_uv_z = (self.rad1-(new_uv**2).sum(-1,True)).sqrt()
            new_uv_points = torch.cat([new_uv, new_uv_z], dim=-1)
            new_st = get_intersect_points_on_sphere(new_uv_points, epi_points, self.rad0)

            new_uv_embeds = self.embedder(new_uv, iter=self.iter)
            new_st_embeds = self.embedder(new_st, iter=self.iter)
        return new_uv_embeds, new_st_embeds

