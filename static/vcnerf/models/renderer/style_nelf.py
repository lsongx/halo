from collections import OrderedDict
import enum
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision.models import vgg16

from mmcv.runner import auto_fp16, force_fp32
from vcnerf.core import im2mse, mse2psnr, raw2outputs, SamplePDF
from ..builder import RENDERER, build_embedder, build_field


class VGGEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp16_enabled = False
        self.vgg16_enc = vgg16(True).features[:17]
        for param in self.vgg16_enc.parameters():
            param.requires_grad = False
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    @auto_fp16()
    def forward(self, x):
        self.vgg16_enc.eval()
        x = F.interpolate(x, size=(224,224), mode='bicubic')
        t = []
        for idx, (m, s) in enumerate(zip(self.mean, self.std)):
            t.append( (x[:,idx]-m)/s )
        t = torch.stack(t, dim=1)
        feats = []
        x = self.vgg16_enc[:4](t) #pool1
        feats.append(x)
        x = self.vgg16_enc[4:10](x) #pool2
        feats.append(x)
        x = self.vgg16_enc[10:17](x) #pool3
        feats.append(x)
        gram_mats = [self.get_gram_mat(feat) for feat in feats]
        return feats, gram_mats

    def get_gram_mat(self, feat):
        b, c, h, w = feat.shape
        F = feat.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G /= h*w
        return G


@RENDERER.register_module()
class StyleNeLF(nn.Module):
    def __init__(self, 
                 embedder, 
                 field, 
                 render_params,):
        super().__init__()
        self.embedder = build_embedder(embedder)
        self.field = build_field(field)

        self.vgg_enc = VGGEnc()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.render_params = render_params
        self.fp16_enabled = False
        self.iter = 0

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

        im_loss = im2mse(outputs['color_map'], rays['rays_color'])
        outputs['rec_loss'] = im_loss

        ori_image = rays['rays_color'].reshape([-1,self.h,self.w,3]).permute([0,3,1,2])
        with torch.no_grad():
            ori_feats, ori_gram = self.vgg_enc(ori_image)
        syn_image = outputs['color_map'].reshape([-1,self.h,self.w,3]).permute([0,3,1,2])
        syn_feats, syn_gram = self.vgg_enc(syn_image)
        aug_syn_image = outputs['aug_color_map'].reshape([-1,self.h,self.w,3]).permute([0,3,1,2])
        aug_syn_feats, aug_syn_gram = self.vgg_enc(aug_syn_image)

        # outputs['content_loss'] = 0
        # # outputs['aug_content_loss'] = 0
        # for o, s, a in zip(ori_feats, syn_feats, aug_syn_feats):
        #     # outputs['content_loss'] += ((o-s).abs()).mean() / len(syn_feats) * 0.01
        #     outputs['aug_content_loss'] += ((
        #         self.global_avg_pool(o)-self.global_avg_pool(a)).abs()).mean() / len(syn_feats) * 0.05

        # # outputs['aug_content_loss'] = ((ori_feats[1]-aug_syn_feats[1]).abs()).mean() / len(aug_syn_feats) * 0.005
        # outputs['aug_content_loss'] = ((ori_feats[1]-aug_syn_feats[1]).abs()).mean() / len(aug_syn_feats) * 0.001

        # # # outputs['gram_loss'] = 0
        # # outputs['aug_gram_loss'] = 0
        # # for o, s, a in zip(ori_gram, syn_gram, aug_syn_gram):
        # #     # outputs['gram_loss'] += ((o-s).abs()).mean() / len(syn_gram) * 0.001
        # #     outputs['aug_gram_loss'] += ((o-a).abs()).mean() / len(aug_syn_gram) * 1e-5
        outputs['aug_gram_loss'] = ((ori_gram[-1]-aug_syn_gram[-1]).abs()).mean() / len(aug_syn_gram) * 0.001

        return outputs

    def _parse_outputs(self, outputs):
        loss, log_vars = self._parse_losses(outputs)
        log_vars['psnr'] = mse2psnr(outputs['rec_loss']).item()
        outputs.update(dict(loss=loss, log_vars=log_vars))
        outputs['num_samples'] = 1
        return outputs

    @auto_fp16()
    def forward_render(self,
                       uv, st, 
                       aug_uv, aug_st, 
                       rays_color, h=400, w=400, batch_ray_forward=False):
        if isinstance(h, torch.Tensor):
            h = h[0]
        self.h, self.w = int(h), int(w)
        b, n, _ = uv.shape
        uvst = torch.cat([uv, st], dim=-1).reshape([-1, 4]) # [b*n,4]
        embeds = self.embedder(uvst)
        if self.training or not batch_ray_forward:
            rgb = self.field(embeds)
        else:
            rgb = self.batch_ray_forward(embeds, batch_ray_forward)
        outputs = {}
        outputs['color_map'] = rgb.reshape([b, n, 3])

        aug_uvst = torch.cat([aug_uv, aug_st], dim=-1).reshape([-1, 4]) # [b*n,4]
        aug_embeds = self.embedder(aug_uvst)
        if self.training or not batch_ray_forward:
            aug_rgb = self.field(aug_embeds)
        else:
            aug_rgb = self.batch_ray_forward(aug_embeds, batch_ray_forward)
        outputs['aug_color_map'] = aug_rgb.reshape([b, n, 3])

        return outputs

    def batch_ray_forward(self, embeds, batch_ray_forward):
        o = []
        i = 0
        while i < embeds.shape[0]:
            end = min(embeds.shape[0], i+batch_ray_forward)
            o.append(self.field(embeds[i:end,...]))
            i += batch_ray_forward
        return torch.cat(o, dim=0)

    def train_step(self, data, optimizer, **kwargs):
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        kwargs['render_params'] = {'batch_ray_forward': 1024}
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

