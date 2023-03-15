import os
import torch
from mmcv import Config
import matplotlib.pyplot as plt
import argparse
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips
import tensorflow as tf
from PIL import Image

from vcnerf.models import build_renderer
from vcnerf.datasets import build_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-f", default="./data/out/")
parser.add_argument("-c", default=None)
args = parser.parse_args()

base = os.path.join(args.f)
for f in os.listdir(base):
    if '.py' in f:
        print(f'{f} loaded.')
        break
cfg_file = f'{base}/{f}'
if args.c is None:
    args.c = f'{base}/latest.pth'

if os.path.isdir(f'{base}/testset-save'):
    os.system(f'rm -rf {base}/testset-save/*')
else:
    os.mkdir(f'{base}/testset-save')

cfg = Config.fromfile(cfg_file)
cfg.data.train.dataset.batch_size = -1

cfg.data.val.split = 'test'
cfg.data.val.testskip = 0
dataset = build_dataset(cfg.data.val)
model = build_renderer(cfg.model).cuda()
state_dict = torch.load(args.c)
if 'state_dict' in state_dict.keys():
    state_dict = state_dict['state_dict']
if 'module' in list(state_dict.keys())[0]:
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # state_dict = new_state_dict
model.load_state_dict(state_dict)
cfg.evaluation.render_params.max_rays_num = 512
# cfg.evaluation.render_params.z_by_nelf = 0.01
# cfg.evaluation.render_params.n_samples = 2

h, w = dataset.h, dataset.w

model.eval()
num_images = len(dataset)
all_psnr = []
all_ssim = []
all_lpips = []
lpips_model = lpips.LPIPS(net='vgg')
for i in range(num_images):
    data = {}
    for k, v in dataset[i].items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda()
        else:
            data[k] = v
    
    with torch.no_grad():
        result = model.forward_render(**data, **cfg.evaluation.render_params)

    nelf_im = result['nelf_color_map'].reshape([h,w,-1]).cpu().numpy().clip(0,1)
    nerf_im = result['nerf_color_map'].reshape([h,w,-1]).cpu().numpy().clip(0,1)
    max_occ = result['nelf_max_occ_map'].reshape([h,w,-1]).cpu().numpy()
    nerf_max_occ = result['nerf_max_occ_map'].reshape([h,w,-1]).cpu().numpy()

    nerf_aug = result.get('nerf_aug_color_map', result['color_map']).reshape([h,w,-1]).cpu().numpy()
    nelf_aug = result.get('nelf_aug_color_map', result['color_map']).reshape([h,w,-1]).cpu().numpy()
    gt = data['rays_color'].cpu().numpy().reshape([h,w,-1]) # extra dim
    
    # plt.imsave(f'{base}/testset-save/gt-{i}.png', gt) # will save 4 channels
    Image.fromarray((gt*255).astype('uint8')).save(f'{base}/testset-save/gt-{i}.png')
    # plt.imsave(f'{base}/testset-save/nerf-{i}.png', nerf_im) # will save 4 channels
    Image.fromarray((nerf_im*255).astype('uint8')).save(f'{base}/testset-save/nerf-{i}.png')

    gt_lpips = data['rays_color'].reshape([h,w,-1]).cpu().permute([2,0,1]) * 2.0 - 1.0
    predict_image_lpips = result['nerf_color_map'].reshape([h,w,-1]).cpu().permute([2,0,1]).clamp(0,1) * 2.0 - 1.0
    lpips_result = lpips_model.forward(predict_image_lpips, gt_lpips).cpu().detach().item()
    all_lpips.append(lpips_result)

    gt_load = tf.image.decode_image(tf.io.read_file(f'{base}/testset-save/gt-{i}.png'))
    pred_load = tf.image.decode_image(tf.io.read_file(f'{base}/testset-save/nerf-{i}.png'))
    gt_load = tf.expand_dims(gt_load, axis=0)
    pred_load = tf.expand_dims(pred_load, axis=0)
    ssim = tf.image.ssim(gt_load, pred_load, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    ssim = float(ssim[0])
    all_ssim.append(ssim)
    # ssim = structural_similarity(gt, nerf_im, win_size=11, multichannel=True, gaussian_weights=True)
    # all_ssim.append(ssim)

    nerf_psnr = peak_signal_noise_ratio(gt, nerf_im, data_range=1)
    nelf_psnr = 0
    if nelf_im.shape[-1] == 3:
        nelf_psnr = peak_signal_noise_ratio(gt, nelf_im, data_range=1)
    print(f'{i:03d}/{num_images:03d} nerf_psnr {nerf_psnr} nelf_psnr {nelf_psnr} lpips_result {lpips_result} ssim {ssim}')
    all_psnr.append([nerf_psnr, nelf_psnr])
all_psnr = torch.tensor(all_psnr)
all_ssim = torch.tensor(all_ssim)
all_lpips = torch.tensor(all_lpips)
print(f'nerf {all_psnr[:,0].mean()} \n nelf {all_psnr[:,1].mean()}')
print(f'ssim {all_ssim.mean()}')
print(f'lpips {all_lpips.mean()}')
