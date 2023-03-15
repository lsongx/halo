from collections import OrderedDict
import enum
from numpy.fft import fftshift
from numpy.lib.arraysetops import isin
import torch
from torch.functional import align_tensors
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.modules import distance

from torchvision.models import vgg16

from mmcv.runner import auto_fp16, force_fp32
from vcnerf.core import im2mse, mse2psnr, raw2outputs, SamplePDF
from ..builder import RENDERER, build_embedder, build_field


@RENDERER.register_module()
class NeLFRangeEPI(nn.Module):
    def __init__(self, 
                 embedder, 
                 field, 
                 epi_reg_weight,
                 grid_epi_weight,
                 grid_range_weight,
                 out_range_ratio,
                 render_params,):
        super().__init__()
        self.embedder = build_embedder(embedder)
        self.field = build_field(field)
        self.epi_reg_weight = epi_reg_weight
        self.grid_epi_weight = grid_epi_weight
        self.grid_range_weight = grid_range_weight
        # self.out_range_ratio = out_range_ratio
        self.render_params = render_params
        self.fp16_enabled = False
        self.iter = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.max_moving_dis = nn.Parameter(torch.tensor([1.,1.]), requires_grad=False)
        self.last_uv = None
        self.last_st = None
        self.last_color = None

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if 'loss' not in loss_name:
                continue
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items())
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self, rays, render_params=None):
        """        
        Args:
            rays (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: 
        """
        if render_params is None:
            render_params = self.render_params
        outputs = self.forward_render(**rays, **render_params)

        return outputs

    def _parse_outputs(self, outputs):
        loss, log_vars = self._parse_losses(outputs)
        log_vars['psnr'] = mse2psnr(outputs['rec_loss']).item()
        outputs.update(dict(loss=loss, log_vars=log_vars))
        outputs['num_samples'] = 1
        return outputs

    @auto_fp16()
    def forward_render(self,
                       uv, st, aug_uv, aug_st, rays_color, 
                       h=400, w=400, batch_ray_forward=False, aug_points=-1):
        if isinstance(h, torch.Tensor):
            h = h[0]
        self.h, self.w = int(h), int(w)
        b, n, _ = uv.shape

        uv_embeds = self.embedder(uv.reshape([-1,2]))
        st_embeds = self.embedder(st.reshape([-1,2]))
        embeds = [uv_embeds, st_embeds]

        if self.training or not batch_ray_forward:
            epi_dir, epi_range, color_code, rgb = self.field(embeds)
        else:
            epi_dir, epi_range, rgb = self.batch_ray_forward(embeds, batch_ray_forward)
            im_loss = torch.tensor(0., device=rgb.device)
            if rays_color is not None:
                im_loss = im2mse(rgb, rays_color)
            return {'color_map': rgb.reshape([b, n, 3]), 
                    'epi_map': torch.atan2(*epi_dir.split(1,1)).reshape([b,n,1]),
                    'range_map': epi_range[:,0].reshape([b,n]),
                    'rec_loss': im_loss}
        outputs = {}
        outputs['color_map'] = rgb.reshape([b, n, 3])
        outputs['epi_map'] = torch.atan2(*epi_dir.split(1,1)).reshape([b,n,1])
        outputs['range_map'] = epi_range[:,0].reshape([b,n])

        im_loss = im2mse(rgb, rays_color)
        outputs['rec_loss'] = im_loss

        if aug_points>0:
            perm = torch.randperm(b*n)
            select_mask = torch.linspace(0,b*n-1,b*n)[perm].long()[:aug_points]
        else:
            select_mask = torch.linspace(0,b*n-1,b*n).long()
            aug_points = b*n

        outputs.update(self.get_epi_move_loss(
            uv[0,select_mask], st[0,select_mask], epi_dir[select_mask], 
            epi_range[select_mask,0], color_code[select_mask], 'ou'))
        outputs.update(self.get_epi_move_loss(
            uv[0,select_mask], st[0,select_mask], epi_dir[select_mask], 
            -epi_range[select_mask,1], color_code[select_mask], 'od'))

        aug_uv_embeds = self.embedder(aug_uv.reshape([-1,2])[select_mask])
        aug_st_embeds = self.embedder(aug_st.reshape([-1,2])[select_mask])
        aug_e, aug_r, aug_c, _ = self.field([aug_uv_embeds, aug_st_embeds])
        outputs.update(self.get_epi_move_loss(
            aug_uv[0,select_mask], aug_st[0,select_mask], aug_e, aug_r[:,0], aug_c, 'au'))
        outputs.update(self.get_epi_move_loss(
            aug_uv[0,select_mask], aug_st[0,select_mask], aug_e, -aug_r[:,1], aug_c, 'ad'))

        outputs.update(self.get_grid_epi_loss(uv, st, epi_dir, epi_range, rays_color))
        self.last_uv = uv
        self.last_st = st
        self.last_color = rays_color

        return outputs

    @auto_fp16()
    def get_epi_move_loss(self, uv, st, ori_e, ori_r, ori_c, label=''):
        outputs = {}
        ori_r = ori_r[:,None]

        # same - in range
        dis = torch.rand([uv.shape[0],1], device=uv.device)
        dis = dis*self.max_moving_dis[None]*ori_r

        us_move = ori_e*dis
        u_move, s_move = us_move[:,0], us_move[:,1]

        us_move_uv = uv
        us_move_uv[:,0] += u_move
        us_move_st = st
        us_move_st[:,0] += s_move
        us_move_uv_embeds = self.embedder(us_move_uv)
        us_move_st_embeds = self.embedder(us_move_st)
        us_move_epi_dir, us_move_range, us_move_color_code, _ = self.field([us_move_uv_embeds, us_move_st_embeds])
        outputs[f'{label}_in_us_epi_loss'] = ((us_move_epi_dir-ori_e)**2).mean()*self.epi_reg_weight
        outputs[f'{label}_in_us_code_loss'] = ((us_move_color_code-ori_c)**2).mean()*self.epi_reg_weight

        vt_move_uv = uv
        vt_move_uv[:,1] += u_move
        vt_move_st = st
        vt_move_st[:,1] += s_move
        vt_move_uv_embeds = self.embedder(vt_move_uv)
        vt_move_st_embeds = self.embedder(vt_move_st)
        vt_move_epi_dir, vt_move_range, vt_move_color_code, _ = self.field([vt_move_uv_embeds, vt_move_st_embeds])
        outputs[f'{label}_in_vt_epi_loss'] = ((vt_move_epi_dir-ori_e)**2).mean()*self.epi_reg_weight
        outputs[f'{label}_in_vt_code_loss'] = ((vt_move_color_code-ori_c)**2).mean()*self.epi_reg_weight


        # # diff - out range
        # dis = torch.rand([uv.shape[0],1], device=uv.device)
        # dis = dis*self.max_moving_dis[None]*self.out_range_ratio + self.max_moving_dis[None]*ori_r
        # us_move = ori_e*dis
        # u_move, s_move = us_move[:,0], us_move[:,1]

        # us_move_uv = uv
        # us_move_uv[:,0] += u_move
        # us_move_st = st
        # us_move_st[:,0] += s_move
        # us_move_uv_embeds = self.embedder(us_move_uv)
        # us_move_st_embeds = self.embedder(us_move_st)
        # us_move_epi_dir, us_move_range, us_move_color_code, _ = self.field([us_move_uv_embeds, us_move_st_embeds])
        # outputs[f'{label}_out_us_epi_loss'] = ((us_move_epi_dir-ori_e)**2).mean()*self.epi_reg_weight
        # outputs[f'{label}_out_us_code_loss'] = ((us_move_color_code-ori_c)**2).mean()*self.epi_reg_weight

        # vt_move_uv = uv
        # vt_move_uv[:,1] += u_move
        # vt_move_st = st
        # vt_move_st[:,1] += s_move
        # vt_move_uv_embeds = self.embedder(vt_move_uv)
        # vt_move_st_embeds = self.embedder(vt_move_st)
        # vt_move_epi_dir, vt_move_range, vt_move_color_code, _ = self.field([vt_move_uv_embeds, vt_move_st_embeds])
        # outputs[f'{label}_out_vt_epi_loss'] = ((vt_move_epi_dir-ori_e)**2).mean()*self.epi_reg_weight
        # outputs[f'{label}_out_vt_code_loss'] = ((vt_move_color_code-ori_c)**2).mean()*self.epi_reg_weight

        return outputs

    @auto_fp16()
    def get_grid_epi_loss(self, uv, st, ori_e, ori_r, gt_c):
        if self.last_uv is None:
            return {'g_rgb_loss': torch.zeros_like(uv), 
                    'g_range_loss': torch.zeros_like(uv)}
        uv_move = (uv-self.last_uv)[0]
        st_move_cal = uv_move/ori_e[:,:1]*ori_e[:,1:]
        new_st = st[0]+st_move_cal
        last_st_min = self.last_st[0].min(0).values
        last_st_max = self.last_st[0].max(0).values
        new_st_coord = ((new_st-last_st_min)/(last_st_max-last_st_min)-1)*2 #[-1,1]

        # inside the last st image
        valid_mask = (new_st_coord<1) & (new_st_coord>-1)
        valid_mask = valid_mask[:,0] | valid_mask[:,1]

        last_img = self.last_color.reshape([1,self.h,self.w,3]).permute([0,3,1,2])
        new_st_coord = new_st_coord[valid_mask][None,None]
        sample_color = torch.nn.functional.grid_sample(last_img, new_st_coord, align_corners=True)

        outputs = {}
        outputs['g_rgb_loss'] = ((sample_color[0,:,0,:].T - gt_c[0,valid_mask])**2).sum(dim=1)
        # ().mean()
        good_rec_mask = outputs['g_rgb_loss'] < 1e-3
        uv_range_max = uv_move.max() / self.max_moving_dis[0]
        uv_range_min = uv_move.min() / self.max_moving_dis[0]
        outputs['g_range_loss'] = torch.zeros_like(uv_range_max)
        if uv_range_min < 0:
            min_loss = ori_r[:,1][valid_mask][good_rec_mask]
            less_than_range_mask = min_loss < -uv_range_min
            outputs['g_range_loss'] += (-uv_range_min-min_loss[less_than_range_mask]).mean()
        if uv_range_max > 0:
            max_loss = ori_r[:,0][valid_mask][good_rec_mask]
            less_than_range_mask = max_loss < uv_range_max
            outputs['g_range_loss'] += (uv_range_max-max_loss[less_than_range_mask]).mean()
        outputs['g_range_loss'] *= self.grid_range_weight
        outputs['g_rgb_loss'] = outputs['g_rgb_loss'].mean()*self.grid_epi_weight
        if torch.isnan(outputs['g_range_loss']):
            outputs['g_range_loss'] = outputs['g_rgb_loss']*0
        return outputs


    @auto_fp16()
    def batch_ray_forward(self, embeds, batch_ray_forward):
        out_e, out_range, out_r = [], [], []
        i = 0
        uv_embeds, st_embeds = embeds
        while i < uv_embeds.shape[0]:
            end = min(uv_embeds.shape[0], i+batch_ray_forward)
            result = self.field([uv_embeds[i:end,...], st_embeds[i:end,...]])
            out_e.append(result[0])
            out_range.append(result[1])
            out_r.append(result[-1])
            i += batch_ray_forward
        return torch.cat(out_e, dim=0), torch.cat(out_range, dim=0), torch.cat(out_r, dim=0)

    def train_step(self, data, optimizer, **kwargs):
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        kwargs['render_params'] = {'batch_ray_forward': 1024}
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

