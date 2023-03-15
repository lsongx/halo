import subprocess
import torch
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-k", type=int, default=0)
parser.add_argument("-g", type=int, default=-1)
parser.add_argument("-t", type=int, default=3)
args = parser.parse_args()
if args.g < 0:
    args.g = args.k
os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.g}'

scenes_train = {
    # 'flower': "['IMG_2968.png', 'IMG_2969.png', 'IMG_2972.png', 'IMG_2970.png']",
    'flower': "['007.png', '008.png', '011.png', '009.png']",
    # 'leaves': "['IMG_3001.png', 'IMG_3002.png', 'IMG_3005.png', 'IMG_3003.png']",
    'leaves': "['004.png', '005.png', '008.png', '006.png']",
    # 'orchids': "['IMG_4474.png', 'IMG_4472.png', 'IMG_4480.png', 'IMG_4481.png']",
    'orchids': "['008.png', '006.png', '014.png', '015.png']",
    'fern': "['005.png', '013.png', '015.png', '016.png']",
    # # 'fortress': "['IMG_1824.png', 'IMG_1825.png', 'IMG_1836.png', 'IMG_1837.png']",
    # 'fortress': "['025.png', '026.png', '037.png', '038.png']",
    # 'horns': "['DJI_20200223_163035_787.png', 'DJI_20200223_163042_402.png', 'DJI_20200223_163101_047.png', 'DJI_20200223_163101_947.png']",
    # 'room': "['DJI_20200226_143944_181.png', 'DJI_20200226_143913_463.png', 'DJI_20200226_143929_618.png', 'DJI_20200226_143930_936.png']",
    # 'trex': "['626.png', '916.png', '755.png', '741.png']",
}

scenes_val = {
    # 'flower': "['IMG_2971.png']",
    'flower': "['010.png']",
    # 'leaves': "['IMG_3004.png']",
    'leaves': "['007.png']",
    # 'orchids': "['IMG_4473.png']",
    'orchids': "['007.png']",
    'fern': "['014.png']",
    # 'fortress': "['IMG_1835.png']",
    'fortress': "['036.png']",
    'horns': "['DJI_20200223_163058_602.png']",
    'room': "['DJI_20200226_143914_572.png']",
    'trex': "['580.png']",
}

while args.k < len(scenes_train):
    scene = list(scenes_train.keys())[args.k]
    if os.path.isdir(f'./data/out/{scene}'):
        os.system(f'rm -r ./data/out/{scene}')

    subprocess.run(f"python -m torch.distributed.launch --nproc_per_node=1 train.py "
                   f"--seed 1 --use_fp16 1 --config ./configs/epi_nelf_sparse_llff_trex.py "
                   f"--work_dir ./data/out/{scene} --port {25888+args.k} "
                   f"--cfg-options data.train.dataset.datadir='~/data/3d/nerf/nerf_llff_data/{scene}' "
                   f"data.train.dataset.select_img='{scenes_train[scene]}' "
                   f"data.val.select_img='{scenes_val[scene]}' "
                   f"data.val.datadir='~/data/3d/nerf/nerf_llff_data/{scene}' ",
                   shell=True)

    for f in os.listdir(f'./data/out/{scene}'):
        if '.log.json' in f:
            with open(f'./data/out/{scene}/{f}', 'r') as tf:
                val_logs = [json.loads(line) for line in tf.read().split('\n') if '"mode": "val"' in line]
            break
    all_psnr = torch.tensor([k['psnr'] for k in val_logs])
    idx = all_psnr.argmax()
    
    if os.path.isfile(f'./data/out/all_llff.json'):
        with open(f'./data/out/all_llff.json', 'r') as f:
            all_test = json.load(f)
    else:
        all_test = {}
    all_test[scene] = {'psnr': float(val_logs[idx]['psnr']), 
                       'ssim': float(val_logs[idx]['ssim']),
                       'lpips': float(val_logs[idx]['lpips'])}
    with open(f'./data/out/all_llff.json', 'w') as f:
        json.dump(all_test, f)
    mean_psnr = torch.tensor(list([k['psnr'] for k in all_test.values()])).mean()
    mean_ssim = torch.tensor(list([k['ssim'] for k in all_test.values()])).mean()
    mean_lpips = torch.tensor(list([k['lpips'] for k in all_test.values()])).mean()
    with open(f'./data/out/mean.log', 'w') as f:
        f.write(f'mean_psnr: {mean_psnr}\n mean_ssim: {mean_ssim}\n mean_lpips: {mean_lpips}')
    args.k += args.t
