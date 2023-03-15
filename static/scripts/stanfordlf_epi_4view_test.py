import torch
import subprocess
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-k", type=int, default=0)
parser.add_argument("-t", type=int, default=3)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.k}'

# large = "['_00_00_', '_00_16_', '_16_00_', '_16_16_']"
large = "['_02_02_', '_02_14_', '_14_02_', '_14_14_']"
add_large = f"data.train.dataset.keep_idx='{large}'"
add_freq = f"model.st_embedder.std=256"
scenes = {
    'cards-small': f"model.pixel_move='[-90,90]' {add_large} {add_freq} ",
    'lego-bulldozer': f"model.pixel_move='[-150,150]' data.train.dataset.non_zero_pretrain=1000 {add_large} {add_freq} ",
    'bracelet': f"model.pixel_move='[-90,90]' {add_large} {add_freq} ",
    'treasure': f"model.pixel_move='[-90,90]' {add_large} {add_freq} ",
    'chess': f"model.pixel_move='[-60,60]' {add_large} {add_freq} ",
    'lego-knights': f"model.pixel_move='[-90,90]' {add_large} {add_freq} ",
    'eucalyptus-flowers': f"model.pixel_move='[-,]' data.train.dataset.non_zero_pretrain=1000 {add_large} {add_freq} ",
    'amethyst': f"model.pixel_move='[-,]' {add_large} {add_freq} ",
    'jellybeans': f"model.pixel_move='[-90,90]' {add_large} {add_freq} ",
    'stanfordbunny': f"model.pixel_move='[-90,90]' {add_large} {add_freq} ",
    'lego-truck': f"model.pixel_move='[-90,90]' {add_large} {add_freq} ",
    'cards-big': f"model.pixel_move='[-190,190]' {add_large} {add_freq} ",
    'lego-gantry': f"model.pixel_move='[-90,90]' {add_large} {add_freq} ",
}

while args.k < len(scenes):
    scene = list(scenes.keys())[args.k]
    if os.path.isdir(f'./data/stanford_epi_4view_test/{scene}'):
        os.system(f'rm -r ./data/stanford_epi_4view_test/{scene}')

    subprocess.run(f"python -m torch.distributed.launch --nproc_per_node=1 train.py "
                   f"--seed 1 --use_fp16 1 --config ./configs/epi_nelf_stanfordlf_sparse_lego_knights.py "
                   f"--work_dir ./data/stanford_epi_4view_test/{scene} --port {25588+args.k} "
                   f"--cfg-options data.train.dataset.base_dir='~/data/3d/StanfordLF/{scene}/rectified' "
                   f"data.val.base_dir='~/data/3d/StanfordLF/{scene}/rectified' "+scenes[scene],
                   shell=True)
    
    for f in os.listdir(f'./data/stanford_epi_4view_test/{scene}'):
        if '.log.json' in f:
            with open(f'./data/stanford_epi_4view_test/{scene}/{f}', 'r') as tf:
                val_logs = [json.loads(line) for line in tf.read().split('\n') if '"mode": "val"' in line]
            break
    all_psnr = torch.tensor([k['psnr'] for k in val_logs])
    idx = all_psnr.argmax()
    
    if os.path.isfile(f'./data/stanford_epi_4view_test/all_stanfordlf.json'):
        with open(f'./data/stanford_epi_4view_test/all_stanfordlf.json', 'r') as f:
            all_test = json.load(f)
    else:
        all_test = {}
    all_test[scene] = {'psnr': float(val_logs[idx]['psnr']), 
                       'ssim': float(val_logs[idx]['ssim']),
                       'lpips': float(val_logs[idx]['lpips'])}
    with open(f'./data/stanford_epi_4view_test/all_stanfordlf.json', 'w') as f:
        json.dump(all_test, f)
    mean_psnr = torch.tensor(list([k['psnr'] for k in all_test.values()])).mean()
    mean_ssim = torch.tensor(list([k['ssim'] for k in all_test.values()])).mean()
    mean_lpips = torch.tensor(list([k['lpips'] for k in all_test.values()])).mean()
    with open(f'./data/stanford_epi_4view_test/mean.log', 'w') as f:
        f.write(f'mean_psnr: {mean_psnr}\n mean_ssim: {mean_ssim}\n mean_lpips: {mean_lpips}')
    args.k += args.t
