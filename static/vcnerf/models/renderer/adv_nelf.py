from collections import OrderedDict
import random
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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp16_enabled = False
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 128, 7, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 1, 1, 0),
        )

    @auto_fp16()
    def forward(self, x):
        return self.discriminator(x)

@RENDERER.register_module()
class AdvNeLF(nn.Module):
    def __init__(self, 
                 embedder, 
                 field, 
                 render_params,):
        super().__init__()
        self.embedder = build_embedder(embedder)
        self.field = build_field(field)

        self.vgg_enc = VGGEnc()
        self.discriminator = Discriminator()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

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
        d_size = render_params.pop('d_size', 128)
        outputs = self.forward_render(**rays, **render_params)
        row = int(random.random()*(self.h-d_size))
        col = int(random.random()*(self.w-d_size))

        im_loss = im2mse(outputs['color_map'], rays['rays_color'])
        outputs['rec_loss'] = im_loss

        aug_fake_image = outputs['aug_color_map'].reshape([-1,self.h,self.w,3]).permute([0,3,1,2])
        real_image = rays['rays_color'].reshape([-1,self.h,self.w,3]).permute([0,3,1,2]).requires_grad_(True)

        with torch.no_grad():
            real_feats, real_gram = self.vgg_enc(real_image)
        aug_syn_feats, aug_syn_gram = self.vgg_enc(aug_fake_image)
        outputs['aug_content_loss'] = ((real_feats[-1]-aug_syn_feats[-1]).abs()).mean() / len(aug_syn_feats) * 0.01

        if self.iter % 2 == 0:
            for p in self.discriminator.parameters():
                p.requires_grad = True
            # dis_real = self.discriminator(real_image[:,:,row:row+d_size,col:col+d_size])
            dis_real = self.discriminator(real_image)
            real_label = torch.ones_like(dis_real)

            aug_fake_image_detached = outputs['aug_color_map'].reshape([-1,self.h,self.w,3]).permute([0,3,1,2]).detach().clone().requires_grad_(True)
            # aug_dis_fake_detached = self.discriminator(aug_fake_image_detached[:,:,row:row+d_size,col:col+d_size])
            aug_dis_fake_detached = self.discriminator(aug_fake_image_detached)
            fake_label = torch.zeros_like(aug_dis_fake_detached)

            outputs['dis_loss'] = (self.bce_loss(dis_real, real_label) + self.bce_loss(aug_dis_fake_detached, fake_label)) * 0.01
            outputs['adv_loss'] = outputs['dis_loss']*0
        else:
            for p in self.discriminator.parameters():
                p.requires_grad = False
            # aug_dis_fake = self.discriminator(aug_fake_image[:,:,row:row+d_size,col:col+d_size])
            aug_dis_fake = self.discriminator(aug_fake_image)
            aug_dis_adv_label = torch.ones_like(aug_dis_fake)
            outputs['adv_loss'] = self.bce_loss(aug_dis_fake, aug_dis_adv_label) * 0.01
            outputs['dis_loss'] = outputs['adv_loss']*0

        # if self.iter>500:
        #     outputs['adv_loss'] = self.bce_loss(aug_dis_fake, aug_dis_adv_label) * 0.01
        # else:
        #     outputs['adv_loss'] = self.bce_loss(aug_dis_fake, aug_dis_adv_label) * 0.0

        # if self.iter<2000000:
        #     fake_image = outputs['color_map'].reshape([-1,400,400,3]).permute([0,3,1,2]).detach().clone().requires_grad_(True)
        #     aug_fake_image = outputs['aug_color_map'].reshape([-1,400,400,3]).permute([0,3,1,2]).detach().clone().requires_grad_(True)
        # elif self.iter % 2:
        #     fake_image = outputs['color_map'].reshape([-1,400,400,3]).permute([0,3,1,2]).detach().clone().requires_grad_(True)
        #     aug_fake_image = outputs['aug_color_map'].reshape([-1,400,400,3]).permute([0,3,1,2])
        # else:
        #     fake_image = outputs['color_map'].reshape([-1,400,400,3]).permute([0,3,1,2])
        #     aug_fake_image = outputs['aug_color_map'].reshape([-1,400,400,3]).permute([0,3,1,2]).detach().clone().requires_grad_(True)
        # dis_fake = self.discriminator(fake_image)
        # aug_dis_fake = self.discriminator(aug_fake_image)
        
        # outputs['real_loss'] = self.bce_loss(dis_real, real_label)
        # if self.iter<2000000:
        #     outputs['fake_loss'] = self.bce_loss(aug_dis_fake, fake_label)+self.bce_loss(dis_fake, fake_label)
        # elif self.iter % 2:
        #     outputs['fake_loss'] = self.bce_loss(aug_dis_fake, real_label)+self.bce_loss(dis_fake, fake_label)
        # else:
        #     outputs['fake_loss'] = self.bce_loss(dis_fake, real_label)+self.bce_loss(aug_dis_fake, fake_label)
        # outputs['fake_loss'] *= 0.5

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
                       rays_color, h=400, w=400
                       ):
        if isinstance(h, torch.Tensor):
            h = h[0]
        self.h, self.w = int(h), int(w)
        b, n, _ = uv.shape
        outputs = {}

        uvst = torch.cat([uv, st], dim=-1).reshape([-1, 4]) # [b*n,4]
        embeds = self.embedder(uvst)
        rgb = self.field(embeds)
        outputs['color_map'] = rgb.reshape([b, n, 3])

        aug_uvst = torch.cat([aug_uv, aug_st], dim=-1).reshape([-1, 4]) # [b*n,4]
        aug_embeds = self.embedder(aug_uvst)
        aug_rgb = self.field(aug_embeds)
        outputs['aug_color_map'] = aug_rgb.reshape([b, n, 3])

        return outputs

    def train_step(self, data, optimizer, **kwargs):
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        return self.train_step(data, optimizer, **kwargs)

