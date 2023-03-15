import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import os.path as osp
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import mmcv
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate


class EvalHook(Hook):

    def __init__(self,
                 dataloader,
                 render_params,
                 extra_log='',
                 interval=1,
                 out_dir=None,
                 logger=None):
        self.dataloader = dataloader
        self.render_params = render_params
        self.interval = interval
        self.out_dir = out_dir
        self.logger = logger
        self.best_psnr = 0
        self.im_shape = (
            int(dataloader.dataset.h), int(dataloader.dataset.w), -1)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        if not self.out_dir:
            self.out_dir = runner.work_dir
        psnr = self.evaluate(runner)
        is_best = False
        if psnr > self.best_psnr:
            is_best = True
            old_filename = f'checkpoint_{self.best_psnr:.2f}.pth'
            if os.path.isfile(osp.join(self.out_dir, old_filename)):
                os.remove(osp.join(self.out_dir, old_filename))
            self.best_psnr = psnr
            self.bestname = f'checkpoint_{self.best_psnr:.2f}.pth'
            if self.logger is not None:
                self.logger.info(f'Saving best {self.bestname}.')
            torch.save(runner.model.state_dict(), 
                       osp.join(self.out_dir, self.bestname))
            torch.save(runner.model.state_dict(), 
                       osp.join(self.out_dir, 'best.pth'))
        else:
            if self.logger is not None:
                self.logger.info(f'Current best {self.bestname}.')

    def evaluate(self, runner):
        runner.model.eval()
        loss = 0
        psnr = 0
        size = 0

        for i, rays in enumerate(self.dataloader):
            with torch.no_grad():
                outputs = runner.model.val_step(rays, 
                                                runner.optimizer,
                                                render_params=self.render_params)

            im = outputs['color_map'].reshape(self.im_shape)
            im = im.detach().cpu().numpy().astype('float')
            gt = rays['rays_color'].reshape(self.im_shape)
            gt = gt.detach().cpu().numpy().astype('float')
            extra = outputs.get(self.extra_log, outputs['color_map'])
            extra = extra.reshape(self.im_shape).detach().cpu().numpy().astype('float')

            fig, axes = plt.subplots(1,3,figsize=(12,4),dpi=256)
            axes[0].imshow(gt)
            axes[1].imshow(im)
            axes[2].imshow(extra)
            save_dir = osp.join(self.out_dir, f'iter{runner.iter+1}-id{i}.png')
            fig.savefig(save_dir, format='png')
            plt.close('all')

            loss += outputs['log_vars']['loss']
            psnr += outputs['log_vars']['psnr']
            size += 1

        loss = loss / size
        psnr = psnr / size
        runner.log_buffer.output['loss'] = loss
        runner.log_buffer.output['PSNR'] = psnr
        runner.log_buffer.ready = True
        return psnr


class DistEvalHook(Hook):

    def __init__(self,
                 dataloader,
                 render_params,
                 extra_log='',
                 save_raw=False,
                 interval=1,
                 out_dir=None,
                 logger=None):
        self.dataloader = dataloader
        self.render_params = render_params
        self.extra_log = extra_log
        self.save_raw = save_raw
        self.interval = interval
        self.out_dir = out_dir
        self.logger = logger
        self.best_psnr = 0
        self.im_shape = (
            int(dataloader.dataset.h), int(dataloader.dataset.w), -1)
        if save_raw:
            import lpips
            import tensorflow as tf
            self.lpips_model = lpips.LPIPS(net='vgg')
            self.tf = tf

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        if not self.out_dir:
            self.out_dir = runner.work_dir
        psnr = self.evaluate(runner)
        if runner.rank == 0:
            is_best = False
            if psnr > self.best_psnr:
                is_best = True
                old_filename = f'checkpoint_{self.best_psnr:.2f}.pth'
                if os.path.isfile(osp.join(self.out_dir, old_filename)):
                    os.remove(osp.join(self.out_dir, old_filename))
                self.best_psnr = psnr
                self.bestname = f'checkpoint_{self.best_psnr:.2f}.pth'
                if self.logger is not None:
                    self.logger.info(f'Saving best {self.bestname}.')
                torch.save(runner.model.state_dict(), 
                           osp.join(self.out_dir, self.bestname))
                torch.save(runner.model.state_dict(), 
                           osp.join(self.out_dir, 'best.pth'))
            else:
                if self.logger is not None:
                    self.logger.info(f'Current best {self.bestname}.')
        dist.barrier()

    def evaluate(self, runner):
        runner.model.eval()
        loss = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
        psnr = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
        size = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
        if self.save_raw:
            ssim_score = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
            lpips_score = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')

        for i, rays in enumerate(self.dataloader):
            with torch.no_grad():
                outputs = runner.model.val_step(rays, 
                                                runner.optimizer,
                                                render_params=self.render_params)

            # save images
            im_ori = outputs['color_map'].reshape(self.im_shape).float()
            im = im_ori.detach().cpu().numpy().astype('float').clip(0,1)
            gt_ori = rays['rays_color'].reshape(self.im_shape).float()
            gt = gt_ori.detach().cpu().numpy().astype('float').clip(0,1)
            extra = outputs.get(self.extra_log, outputs['color_map'])
            extra = extra.reshape(self.im_shape).detach().cpu().numpy().astype('float')

            if self.save_raw:
                save_dir = osp.join(self.out_dir, 
                                    f'iter{runner.iter+1}-epoch{runner.epoch+1}'
                                    f'-id{runner.rank+i}')
                Image.fromarray((255*gt).astype('uint8')).save(save_dir+'-gt.png')
                Image.fromarray((255*im).astype('uint8')).save(save_dir+'-im.png')
                fig, axes = plt.subplots(1,1,figsize=(4,4),dpi=256)
                axes.imshow(extra)
                fig.savefig(save_dir+'-extra.png', format='png')
                plt.close('all')
                # plt.imsave(save_dir+'-extra.png', extra[:,:,0])

                gt_lpips = gt_ori.cpu().permute([2,0,1]) * 2.0 - 1.0
                predict_image_lpips = im_ori.cpu().permute([2,0,1]).clamp(0,1) * 2.0 - 1.0
                lpips_score += self.lpips_model.forward(predict_image_lpips, gt_lpips).cpu().detach().item()

                gt_load = self.tf.image.decode_image(self.tf.io.read_file(save_dir+'-gt.png'))
                pred_load = self.tf.image.decode_image(self.tf.io.read_file(save_dir+'-im.png'))
                gt_load = self.tf.expand_dims(gt_load, axis=0)
                pred_load = self.tf.expand_dims(pred_load, axis=0)
                ssim = self.tf.image.ssim(gt_load, pred_load, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
                ssim_score += float(ssim[0])
            else:
                fig, axes = plt.subplots(1,3,figsize=(12,4),dpi=256)
                axes[0].imshow(gt)
                axes[1].imshow(im)
                axes[2].imshow(extra)
                save_dir = osp.join(self.out_dir, 
                                    f'iter{runner.iter+1}-epoch{runner.epoch+1}'
                                    f'-id{runner.rank+i}.png')
                fig.savefig(save_dir, format='png')
                plt.close('all')

            loss += outputs['log_vars']['loss']
            psnr += outputs['log_vars']['psnr']
            size += 1

        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(psnr, op=dist.ReduceOp.SUM)
        dist.all_reduce(size, op=dist.ReduceOp.SUM)
        loss = loss.item()/size.item()
        psnr = psnr.item()/size.item()
        runner.log_buffer.output['loss'] = loss
        runner.log_buffer.output['psnr'] = psnr
        if self.save_raw:
            dist.all_reduce(ssim_score, op=dist.ReduceOp.SUM)
            dist.all_reduce(lpips_score, op=dist.ReduceOp.SUM)
            runner.log_buffer.output['ssim'] = ssim_score.item()/size.item()
            runner.log_buffer.output['lpips'] = lpips_score.item()/size.item()
        runner.log_buffer.ready = True
        return psnr


import matplotlib.pyplot as plt
class DistEvalHookWithDepth(DistEvalHook):

    def evaluate(self, runner):
        runner.model.eval()
        loss = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
        psnr = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
        size = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')

        for i, rays in enumerate(self.dataloader):
            with torch.no_grad():
                outputs = runner.model.val_step(rays, 
                                                runner.optimizer,
                                                render_params=self.render_params)

            # save images
            im = outputs['coarse']['color_map'].reshape(self.im_shape)
            im = im.detach().cpu().numpy()
            spn_depth = outputs['coarse']['spn_depth'].reshape(self.im_shape[:2]).cpu().numpy().astype('float32')
            pred_depth = outputs['coarse']['depth_map'].reshape(self.im_shape[:2]).cpu().numpy().astype('float32')
            fig, axes = plt.subplots(1, 3, figsize=(19, 6))
            axes[0].imshow(im)
            spn = axes[1].imshow(spn_depth)
            pred = axes[2].imshow(pred_depth)
            fig.colorbar(spn, ax=axes[1])
            fig.colorbar(pred, ax=axes[2])
            fig.savefig(osp.join(self.out_dir, f'iter{runner.iter+1}-id{runner.rank+i}.png'), format='png')

            loss += outputs['log_vars']['loss']
            psnr += outputs['log_vars']['psnr']
            size += 1

        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(psnr, op=dist.ReduceOp.SUM)
        dist.all_reduce(size, op=dist.ReduceOp.SUM)
        loss = loss.item()/size.item()
        psnr = psnr.item()/size.item()
        runner.log_buffer.output['loss'] = loss
        runner.log_buffer.output['PSNR'] = psnr
        runner.log_buffer.ready = True
        return psnr



class DistMPDEvalHook(DistEvalHook):
    def __init__(self,
                 dataloader,
                 render_params,
                 interval=1,
                 out_dir=None,
                 logger=None):
        self.dataloader = dataloader
        self.render_params = render_params
        self.interval = interval
        self.out_dir = out_dir
        self.logger = logger
        self.best_iou = 0

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        if not self.out_dir:
            self.out_dir = runner.work_dir
        iou = self.evaluate(runner)
        if runner.rank == 0:
            if iou > self.best_iou:
                old_filename = f'checkpoint_{self.best_iou:.2f}.pth'
                if os.path.isfile(osp.join(self.out_dir, old_filename)):
                    os.remove(osp.join(self.out_dir, old_filename))
                self.best_iou = iou
                self.bestname = f'checkpoint_{self.best_iou:.2f}.pth'
                if self.logger is not None:
                    self.logger.info(f'Saving best {self.bestname}.')
                torch.save(runner.model.state_dict(), 
                           osp.join(self.out_dir, self.bestname))
        dist.barrier()

    def evaluate(self, runner):
        runner.model.eval()
        loss = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
        size = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')

        for i, data in enumerate(self.dataloader):
            with torch.no_grad():
                outputs = runner.model.val_step(data, 
                                                runner.optimizer,
                                                render_params=self.render_params)

            # save images
            gt = data['img'][0].numpy()
            im = outputs['coarse']['color_map'].reshape(gt.shape)
            im = im.detach().cpu().numpy()
            if i == 0:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                axes[0].imshow(im)
                axes[1].imshow(gt)
                fig.savefig(osp.join(self.out_dir, f'iter{runner.iter+1}-id{runner.rank+i}.png'), format='png')

            loss += ((im-gt)**2).mean()
            size += 1

        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(size, op=dist.ReduceOp.SUM)
        loss = loss.item()/size.item()
        runner.log_buffer.output['loss'] = loss
        runner.log_buffer.ready = True
        return loss