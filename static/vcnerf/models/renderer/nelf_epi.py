from collections import OrderedDict
import enum
from numpy.fft import fftshift
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.modules import distance

from torchvision.models import vgg16

from mmcv.runner import auto_fp16, force_fp32
from vcnerf.core import im2mse, mse2psnr, raw2outputs, SamplePDF
from ..builder import RENDERER, build_embedder, build_field


@RENDERER.register_module()
class NeLFEPI(nn.Module):
    def __init__(self, 
                 embedder, 
                 field, 
                 epi_reg_weight,
                 epi_smooth_weight,
                 grid_proj_weight,
                 epi_larger_remove_iter,
                 init_epi_range,
                 min_epi_range,
                 re_init_start,
                 re_init_range,
                 render_params,):
        super().__init__()
        self.embedder = build_embedder(embedder)
        self.field = build_field(field)
        self.epi_reg_weight = epi_reg_weight
        self.epi_smooth_weight = epi_smooth_weight
        self.grid_proj_weight = grid_proj_weight
        self.epi_larger_remove_iter = epi_larger_remove_iter
        self.init_epi_range = init_epi_range
        self.min_epi_range = min_epi_range
        self.re_init_start = re_init_start
        self.re_init_range = re_init_range
        self.render_params = render_params
        self.fp16_enabled = False
        self.iter = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.max_moving_dis = nn.Parameter(torch.tensor([1.,1.]), requires_grad=False)

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
        self.uv = uv
        self.st = st
        self.rays_color = rays_color
        b, n, _ = uv.shape

        uv_embeds = self.embedder(uv.reshape([-1,2]))
        st_embeds = self.embedder(st.reshape([-1,2]))
        embeds = [uv_embeds, st_embeds]

        if self.training or not batch_ray_forward:
            epi_dir, color_code, rgb = self.field(embeds)
        else:
            epi_dir, rgb = self.batch_ray_forward(embeds, batch_ray_forward)
            im_loss = torch.tensor(0., device=rgb.device)
            if rays_color is not None:
                im_loss = im2mse(rgb, rays_color)
            return {'color_map': rgb.reshape([b,n,3]), 
                    'epi_map': epi_dir.reshape([b,n,1]),
                    'rec_loss': im_loss}
        outputs = {}
        outputs['color_map'] = rgb.reshape([b,n,3])
        outputs['epi_map'] = epi_dir.reshape([b,n,1])

        im_loss = im2mse(rgb, rays_color)
        outputs['rec_loss'] = im_loss

        if aug_points>0:
            perm = torch.randperm(b*n)
            select_mask = torch.linspace(0,b*n-1,b*n)[perm].long()[:aug_points]
        else:
            select_mask = torch.linspace(0,b*n-1,b*n).long()
            aug_points = b*n

        outputs.update(self.get_epi_move_loss(
            uv[0,select_mask], st[0,select_mask], 
            epi_dir[select_mask], color_code[select_mask]))

        return_zero = False
        if self.is_re_init(): # not re init iters
            return_zero = True

        aug_uv_embeds = self.embedder(aug_uv.reshape([-1,2])[select_mask])
        aug_st_embeds = self.embedder(aug_st.reshape([-1,2])[select_mask])
        aug_e, aug_c, _ = self.field([aug_uv_embeds, aug_st_embeds])
        outputs.update(self.get_epi_move_loss(
            aug_uv[0,select_mask], aug_st[0,select_mask], aug_e, aug_c, return_zero, 'aug'))

        outputs.update(self.get_epi_smooth_loss(epi_dir, rays_color))
        outputs.update(self.get_epi_re_init_loss(
            uv[0,select_mask], st[0,select_mask], epi_dir[select_mask]))
        return outputs

    def get_epi_move_loss(self, uv, st, ori_e, ori_c, return_zero=False, label=''):
        if return_zero:
            return {f'{label}epi_loss': torch.tensor(0., device=uv.device),
                    f'{label}code_loss': torch.tensor(0., device=uv.device),}
                    # f'{label}proj_loss': torch.tensor(0., device=uv.device),}
        outputs = {}
        aug_points = uv.shape[0]

        progress = self.iter/max(self.epi_larger_remove_iter, self.iter)
        epi_range = self.init_epi_range*(1-progress) + self.min_epi_range*progress
        us_dis = torch.rand([aug_points,1], device=uv.device)-0.5
        us_dis = us_dis*2*self.max_moving_dis.max()*epi_range
        u_move, s_move = us_dis*ori_e.cos(), us_dis*ori_e.sin()
        vt_dis = torch.rand([aug_points,1], device=uv.device)-0.5
        vt_dis = vt_dis*2*self.max_moving_dis.max()*epi_range
        v_move, t_move = vt_dis*ori_e.cos(), vt_dis*ori_e.sin()

        move_uv = uv + torch.cat([u_move, v_move], dim=1)
        move_st = st + torch.cat([s_move, t_move], dim=1)
        move_uv_embeds = self.embedder(move_uv)
        move_st_embeds = self.embedder(move_st)
        move_epi_dir, move_color_code, move_rgb = self.field([move_uv_embeds, move_st_embeds])
        outputs[f'{label}epi_loss'] = ((move_epi_dir-ori_e)**2)*self.epi_reg_weight
        outputs[f'{label}code_loss'] = ((move_color_code-ori_c)**2)*self.epi_reg_weight
        # outputs[f'{label}proj_loss'] = torch.tensor(0., device=uv.device)
        if self.iter > self.epi_larger_remove_iter:
            smaller_mask = (move_epi_dir<ori_e)[:,0]
            outputs[f'{label}epi_loss'] = outputs[f'{label}epi_loss'][smaller_mask].mean()
            outputs[f'{label}code_loss'] = outputs[f'{label}code_loss'][smaller_mask].mean()
            # if (~smaller_mask).sum() > 0:
            #     outputs[f'{label}proj_loss'] = self.get_grid_proj_loss(
            #         move_uv[~smaller_mask], move_st[~smaller_mask], 
            #         move_epi_dir[~smaller_mask], move_color_code[~smaller_mask])

        return outputs

    def get_epi_re_init_loss(self, uv, st, ori_e):
        outputs = {}
        outputs[f'epi_re_init_loss'] = torch.tensor(0., device=uv.device)
        if self.is_re_init(): # not re init iters
            return outputs

        aug_points = uv.shape[0]
        epi_range = self.init_epi_range # largest range to re init
        us_dis = torch.rand([aug_points,1], device=uv.device)-0.5
        us_dis = us_dis*2*self.max_moving_dis.max()*epi_range
        u_move, s_move = us_dis*ori_e.cos(), us_dis*ori_e.sin()
        vt_dis = torch.rand([aug_points,1], device=uv.device)-0.5
        vt_dis = vt_dis*2*self.max_moving_dis.max()*epi_range
        v_move, t_move = vt_dis*ori_e.cos(), vt_dis*ori_e.sin()

        move_uv = uv + torch.cat([u_move, v_move], dim=1)
        move_st = st + torch.cat([s_move, t_move], dim=1)
        move_uv_embeds = self.embedder(move_uv)
        move_st_embeds = self.embedder(move_st)
        move_epi_dir, _, _ = self.field([move_uv_embeds, move_st_embeds])

        smaller_mask = move_epi_dir < ori_e # aug move to a smaller, propagate, no loss to ori
        outputs[f'epi_re_init_loss'] += ((move_epi_dir[smaller_mask]-ori_e[smaller_mask].detach())**2).mean()*self.epi_reg_weight

        larger_mask = move_epi_dir > ori_e # aug move to a larger ray, re init it to zero
        outputs[f'epi_re_init_loss'] += ((ori_e[larger_mask])**2).mean()*self.epi_reg_weight
        return outputs

    def is_re_init(self):
        for s in self.re_init_start:
            dif = self.iter - s
            if dif > 0 and dif < self.re_init_range:
                return True
        return False

    def get_epi_smooth_loss(self, epi, color):
        epi = epi.reshape([self.h,self.w])
        color = color.reshape([self.h,self.w,3])
        grad_disp_x = torch.abs(epi[:, :-1] - epi[:, 1:])
        grad_disp_y = torch.abs(epi[:-1, :] - epi[1:, :])

        grad_img_x = torch.mean(torch.abs(color[:, :-1, :] - color[:, 1:, :]), 2)
        grad_img_y = torch.mean(torch.abs(color[:-1, :, :] - color[1:, :, :]), 2)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)
        smooth_loss = (grad_disp_x.mean() + grad_disp_y.mean())*self.epi_smooth_weight
        return {'epi_smooth_loss': smooth_loss}

    def get_grid_proj_loss(self, uv_move, st_move, move_epi, move_color_code):
        move_distance = (uv_move-self.uv[0,:uv_move.shape[0],:])/(move_epi.cos()+1e-5)
        back_proj_st = st_move-move_distance*move_epi.sin()
        st_min = self.st[0].min(0).values
        st_max = self.st[0].max(0).values
        new_st_coord = ((back_proj_st-st_min)/(st_max-st_min)-1)*2 #[-1,1]

        valid_mask = (new_st_coord<1) & (new_st_coord>-1)
        valid_mask = valid_mask[:,0] & valid_mask[:,1]
        if valid_mask.sum() == 0:
            return torch.zeros_like(st_min)

        img = self.rays_color.reshape([1,self.h,self.w,3]).permute([0,3,1,2])
        new_st_coord = new_st_coord[valid_mask][None,None]
        sample_color = torch.nn.functional.grid_sample(img, new_st_coord, align_corners=True)
        sample_color = sample_color[0,:,0,:].T

        with torch.no_grad():
            uv_embeds = self.embedder(self.uv[0])[:valid_mask.sum()]
            move_color = self.field.forward_with_color_code(move_color_code[valid_mask], uv_embeds)
        color_loss = ((move_color-sample_color)**2).mean()*self.grid_proj_weight
        # color_loss = ((move_color_code[valid_mask]-sample_color)**2).mean()*self.grid_proj_weight
        return color_loss

    def batch_ray_forward(self, embeds, batch_ray_forward):
        out_e, out_r = [], []
        i = 0
        uv_embeds, st_embeds = embeds
        while i < uv_embeds.shape[0]:
            end = min(uv_embeds.shape[0], i+batch_ray_forward)
            result = self.field([uv_embeds[i:end,...], st_embeds[i:end,...]])
            out_e.append(result[0])
            out_r.append(result[2])
            i += batch_ray_forward
        return torch.cat(out_e, dim=0), torch.cat(out_r, dim=0)

    def train_step(self, data, optimizer, **kwargs):
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        kwargs['render_params'] = {'batch_ray_forward': 1024}
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

