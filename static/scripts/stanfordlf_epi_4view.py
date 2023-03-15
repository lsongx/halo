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

# scenes = {
#     'chess': "model.pixel_move='[-50,50]' ",
#     'lego-knights': "model.pixel_move='[-75,75]' ",
#     'cards-big': "model.pixel_move='[-200,200]' ",
#     'cards-small': "model.pixel_move='[-75,75]' ",
#     'eucalyptus-flowers': "model.pixel_move='[-35,35]' data.train.dataset.non_zero_pretrain=1000 ",
#     'lego-bulldozer': "model.pixel_move='[-120,120]' ",
#     'treasure': "model.pixel_move='[-75,75]' data.train.dataset.non_zero_pretrain=1000 ",
#     'amethyst': "model.pixel_move='[-35,35]' ",
#     'bracelet': "model.pixel_move='[-75,75]' ",
#     'jellybeans': "model.pixel_move='[-75,75]' ",
#     'lego-truck': "model.pixel_move='[-75,75]' ",
#     'stanfordbunny': "model.pixel_move='[-75,75]' ",
#     'lego-gantry': "model.pixel_move='[-75,75]' ",
# }
scenes = {
    'eucalyptus-flowers': "model.pixel_move='[-25,25]' data.train.dataset.non_zero_pretrain=1000 ",
    'lego-bulldozer': "model.pixel_move='[-100,100]' ",
    'treasure': "model.pixel_move='[-50,50]' data.train.dataset.non_zero_pretrain=1000 ",
    'amethyst': "model.pixel_move='[-25,25]' ",
    'bracelet': "model.pixel_move='[-50,50]' ",
    'jellybeans': "model.pixel_move='[-50,50]' ",
    'lego-truck': "model.pixel_move='[-50,50]' ",
    'stanfordbunny': "model.pixel_move='[-50,50]' ",
    'chess': "model.pixel_move='[-35,35]' ",
    'lego-knights': "model.pixel_move='[-50,50]' ",
    'cards-big': "model.pixel_move='[-150,150]' ",
    'cards-small': "model.pixel_move='[-50,50]' ",
    'lego-gantry': "model.pixel_move='[-50,50]' ",
}

while args.k < len(scenes):
    scene = list(scenes.keys())[args.k]
    if os.path.isdir(f'./data/stanford_epi_4view/{scene}'):
        os.system(f'rm -r ./data/stanford_epi_4view/{scene}')

    subprocess.run(f"python -m torch.distributed.launch --nproc_per_node=1 train.py "
                   f"--seed 1 --use_fp16 1 --config ./configs/epi_nelf_stanfordlf_sparse_lego_knights.py "
                   f"--work_dir ./data/stanford_epi_4view/{scene} --port {25588+args.k} "
                   f"--cfg-options data.train.dataset.base_dir='~/data/3d/StanfordLF/{scene}/rectified' "
                   f"data.val.base_dir='~/data/3d/StanfordLF/{scene}/rectified' "+scenes[scene],
                   shell=True)
    
    for f in os.listdir(f'./data/stanford_epi_4view/{scene}'):
        if '.log.json' in f:
            with open(f'./data/stanford_epi_4view/{scene}/{f}', 'r') as tf:
                val_logs = [json.loads(line) for line in tf.read().split('\n') if '"mode": "val"' in line]
            break
    all_psnr = torch.tensor([k['psnr'] for k in val_logs])
    idx = all_psnr.argmax()
    
    if os.path.isfile(f'./data/stanford_epi_4view/all_stanfordlf.json'):
        with open(f'./data/stanford_epi_4view/all_stanfordlf.json', 'r') as f:
            all_test = json.load(f)
    else:
        all_test = {}
    all_test[scene] = {'psnr': float(val_logs[idx]['psnr']), 
                       'ssim': float(val_logs[idx]['ssim']),
                       'lpips': float(val_logs[idx]['lpips'])}
    with open(f'./data/stanford_epi_4view/all_stanfordlf.json', 'w') as f:
        json.dump(all_test, f)
    mean_psnr = torch.tensor(list([k['psnr'] for k in all_test.values()])).mean()
    mean_ssim = torch.tensor(list([k['ssim'] for k in all_test.values()])).mean()
    mean_lpips = torch.tensor(list([k['lpips'] for k in all_test.values()])).mean()
    with open(f'./data/stanford_epi_4view/mean.log', 'w') as f:
        f.write(f'mean_psnr: {mean_psnr}\n mean_ssim: {mean_ssim}\n mean_lpips: {mean_lpips}')
    args.k += args.t
