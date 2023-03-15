# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='NeLFNeRF360Joint',
    xyz_embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        nb_freqs=10, 
        include_input=False),
    lf_xyz_embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        nb_freqs=5, 
        scale=2**3,
        include_input=False),
    uv_embedder=dict(
        type='NormalEmbedder',
        in_dims=3,
        nb_freqs=10,
        std=16,
        scale=8,
        include_input=False),
    st_embedder=dict(
        type='NormalEmbedder',
        in_dims=3, 
        nb_freqs=10, 
        std=16, 
        include_input=False),
    dir_embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        nb_freqs=4, 
        include_input=False),
    coarse_field=dict(
        type='BaseField',
        nb_layers=8, 
        hid_dims=256, 
        xyz_emb_dims=2*3*10,
        dir_emb_dims=2*3*4,
        use_dirs=False),
    nelf_field=dict(
        type='NeLFEPIField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=2*3*(10),#+2,
        color_uv_dims=0,#+2,
        use_sin=False),
    render_params=dict( # default render cfg; train cfg
        n_samples=128,
        perturb=True,
        alpha_noise_std=1.0,
        inv_depth=False,
        use_dirs=True,
        background='white',
        z_by_nelf=0.2,
        max_rays_num=1024,),
    rec_loss_weight=1,
    consist_loss_weight=1,
    nelf_converge_iter=5e3)

# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='AugDTUDataset',
            datadir='~/data/3d/rs_dtu_4', 
            load_object='scan21',
            batch_size=1024*3,
            select_idx=[22,25,28],
            batching=True,
            to_cuda=True),
        times=15),
    val=dict(
        type='DTUDataset',
        datadir='~/data/3d/rs_dtu_4', 
        load_object='scan21',
        batch_size=-1,
        exclude_idx=[22,25,28],
        batching=False,
        to_cuda=True),)

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='Exp', gamma=0.1**(1/50000), by_epoch=False) 
# lr_config = dict(policy='Step', step=[20,40,60,80], gamma=0.5, by_epoch=True)
# runner = dict(type='EpochBasedRunner', max_epochs=100)
lr_config = dict(policy='Poly', power=2, min_lr=5e-6, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=50)
# misc settings
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', interval_exp_name=5000),
        # dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(
    interval=1,
    save_raw=True,
    extra_log='nerf_depth_map',
    render_params=dict(
        n_samples=128,
        # n_samples=128+64,
        perturb=False,
        alpha_noise_std=0,
        inv_depth=False,
        use_dirs=True,
        z_by_nelf=0.3,
        background='white',
        max_rays_num=1024*3,))
extra_hooks = [dict(type='IterAdjustHook',),]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
