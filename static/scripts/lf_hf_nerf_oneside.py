import subprocess
import os
import json

scene = {    
    'oneside-lego': "['r_58.png','r_5.png','r_2.png','r_8.png','r_9.png','r_10.png','r_16.png','r_34.png','r_35.png','r_40.png','r_52.png','r_53.png','r_54.png','r_60.png',]",
}

param = {
    'oneside-lego': '',
}

for k, v in scene.items():
    scene_name = k.split('-')[1]
    if not os.path.isfile(f'./data/out/{k}-lfnerf/epoch_3.pth'):
        subprocess.run(f"python -m torch.distributed.launch --nproc_per_node=1 train.py "
                       f"--seed 1 --use_fp16 1 --config ./configs/nelf_nerf_synthetic_hotdog_sparse_lfnerf.py "
                       f"--work_dir ./data/out/{k}-lfnerf "
                       f"--cfg-options data.train.dataset.base_dir='~/data/3d/nerf/nerf_synthetic/{scene_name}' "
                       f'data.train.dataset.select_imgs="{v}" '
                       f"model.xyz_embedder.scale='1' "
                       f"runner.max_epochs='3' "
                       f"data.val.base_dir='~/data/3d/nerf/nerf_synthetic/{scene_name}' ",
                       shell=True)
    if not os.path.isfile(f'./data/out/{k}-nelf/epoch_30.pth'):
        subprocess.run(f"python -m torch.distributed.launch --nproc_per_node=1 train.py "
                       f"--seed 1 --use_fp16 1 --config ./configs/nelf_nerf_synthetic_hotdog_sparse_nelf.py "
                       f"--work_dir ./data/out/{k}-nelf "
                       f"--cfg-options data.train.dataset.base_dir='~/data/3d/nerf/nerf_synthetic/{scene_name}' "
                       f'data.train.dataset.select_imgs="{v}" '
                       f"data.val.base_dir='~/data/3d/nerf/nerf_synthetic/{scene_name}' "
                       f"model.xyz_embedder.scale='1' "
                       f"model.nerf_pretrain='./data/out/{k}-lfnerf/epoch_3.pth' ",
                       shell=True)
    subprocess.run(f"python -m torch.distributed.launch --nproc_per_node=1 train.py "
                   f"--seed 1 --use_fp16 1 --config ./configs/nelf_nerf_synthetic_hotdog_sparse_hfnerf.py "
                   f"--work_dir ./data/out/{k}-hfnerf "
                   f"--cfg-options data.train.dataset.base_dir='~/data/3d/nerf/nerf_synthetic/{scene_name}' "
                   f"model.lf_xyz_embedder.scale='1' "
                   f"model.empty_loss_weight='0.01' "
                   f"model.hf_load_lf='True' optimizer.lr='5e-3' runner.max_epochs='10' "
                   f'data.train.dataset.select_imgs="{v}" '
                   f"data.val.base_dir='~/data/3d/nerf/nerf_synthetic/{scene_name}' "
                   f"model.nerf_pretrain='./data/out/{k}-lfnerf/epoch_3.pth' "
                   f"model.nelf_pretrain='./data/out/{k}-nelf/epoch_30.pth' "+param[k],
                   shell=True)
    exp_return = subprocess.run(f"python nerf_synthetic_testset.py "
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
