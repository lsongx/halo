import subprocess
import os
import json

scene = {    
    'lego': "['r_2.png', 'r_16.png', 'r_93.png', 'r_55.png', 'r_73.png', 'r_86.png', 'r_26.png', 'r_75.png']",
    'chair': "['r_86.png', 'r_73.png', 'r_26.png', 'r_2.png', 'r_55.png', 'r_93.png', 'r_16.png', 'r_75.png']",
    'drums': "['r_86.png', 'r_93.png', 'r_75.png', 'r_26.png', 'r_55.png', 'r_73.png', 'r_16.png', 'r_2.png']",
    # 'ficus': "['r_2.png', 'r_93.png', 'r_73.png', 'r_86.png', 'r_75.png', 'r_26.png', 'r_55.png', 'r_16.png']",
    # 'mic': "['r_55.png', 'r_2.png', 'r_93.png', 'r_75.png', 'r_16.png', 'r_26.png', 'r_86.png', 'r_73.png']",
    # 'ship': "['r_55.png', 'r_93.png', 'r_26.png', 'r_75.png', 'r_16.png', 'r_33.png', 'r_73.png', 'r_86.png']",
    # 'materials': "['r_75.png', 'r_26.png', 'r_93.png', 'r_55.png', 'r_86.png', 'r_16.png', 'r_2.png', 'r_73.png']",
    # 'hotdog': "['r_16.png', 'r_93.png', 'r_75.png', 'r_86.png', 'r_2.png', 'r_55.png', 'r_73.png', 'r_26.png']",
}

param = {
    'lego': "model.empty_start_iter='0' model.hf_load_lf='True' optimizer.lr='5e-3' lr_config.power='1' ",
    'chair': "model.empty_start_iter='0' lr_config.power='1' ",
    'drums': "model.empty_start_iter='2e3'  ",
    'ficus': "model.empty_start_iter='2e3'  ",
    'mic': "model.empty_start_iter='2e3' ",
    'ship': "model.empty_start_iter='0' ",
    'materials': "model.empty_start_iter='0' ",
    'hotdog': "model.empty_start_iter='0' model.hf_load_lf='True' optimizer.lr='5e-3' lr_config.power='1' ",
}

if __name__ == '__main__':
    for k, v in scene.items():
        if not os.path.isfile(f'./data/out/{k}-lfnerf/epoch_10.pth'):
            subprocess.run(f"CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py "
                           f"--seed 1 --use_fp16 1 --config ./configs/nelf_nerf_synthetic_hotdog_sparse_lfnerf.py "
                           f"--work_dir ./data/out/{k}-lfnerf "
                           f"--cfg-options data.train.dataset.base_dir='~/data/3d/nerf/nerf_synthetic/{k}' "
                           f'data.train.dataset.select_imgs="{v}" '
                           f"data.val.base_dir='~/data/3d/nerf/nerf_synthetic/{k}' ",
                           shell=True)
        if not os.path.isfile(f'./data/out/{k}-nelf/epoch_30.pth'):
            subprocess.run(f"CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py "
                           f"--seed 1 --use_fp16 1 --config ./configs/nelf_nerf_synthetic_hotdog_sparse_nelf.py "
                           f"--work_dir ./data/out/{k}-nelf "
                           f"--cfg-options data.train.dataset.base_dir='~/data/3d/nerf/nerf_synthetic/{k}' "
                           f'data.train.dataset.select_imgs="{v}" '
                           f"data.val.base_dir='~/data/3d/nerf/nerf_synthetic/{k}' "
                           f"model.nerf_pretrain='./data/out/{k}-lfnerf/epoch_10.pth' ",
                           shell=True)
        subprocess.run(f"CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py "
                       f"--seed 1 --use_fp16 1 --config ./configs/nelf_nerf_synthetic_hotdog_sparse_hfnerf.py "
                       f"--work_dir ./data/out/{k}-hfnerf "
                       f"--cfg-options data.train.dataset.base_dir='~/data/3d/nerf/nerf_synthetic/{k}' "
                       f'data.train.dataset.select_imgs="{v}" '
                       f"data.val.base_dir='~/data/3d/nerf/nerf_synthetic/{k}' "
                       f"model.nerf_pretrain='./data/out/{k}-lfnerf/epoch_10.pth' "
                       f"model.nelf_pretrain='./data/out/{k}-nelf/epoch_30.pth' "+param[k],
                       shell=True)
        exp_return = subprocess.run(f"CUDA_VISIBLE_DEVICES=0 python nerf_synthetic_testset.py "
                                    f"-f ./data/out/{k}-hfnerf "
                                    f"-c ./data/out/{k}-hfnerf/best.pth",
                                    shell=True, capture_output=True)
        with open(f'./data/out/run_{k}.txt', 'w') as f:
            f.write(f"{exp_return.stdout.decode('ascii')}")
        test = exp_return.stdout.decode('ascii').split('\nnerf ')[-1][:5]
        if os.path.isfile(f'./data/out/all.json'):
            with open(f'./data/out/all.json', 'r') as f:
                all_test = json.load(f)
        else:
            all_test = {}
        all_test[k] = test
        with open(f'./data/out/all.json', 'w') as f:
            json.dump(all_test, f)