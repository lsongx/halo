# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='NeLFNeRF',
    xyz_embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        nb_freqs=10, 
        include_input=False),
    # xyz_embedder=dict(
    #     type='NormalEmbedder',
    #     in_dims=3, 
    #     nb_freqs=10, 
    #     std=64,
    #     scale=1,
    #     include_input=False),
    uv_embedder=dict(
        type='NormalEmbedder',
        in_dims=2, 
        nb_freqs=10,
        std=16,
        scale=0.1,
        include_input=False),
    # st_embedder=dict(
    #     type='NormalEmbedder',
    #     in_dims=2, 
    #     nb_freqs=10,
    #     std=16,
    #     scale=1,
    #     include_input=False),
    # uv_embedder=dict(
    #     type='BaseEmbedder',
    #     in_dims=2, 
    #     nb_freqs=10, 
    #     scale=1,
    #     include_input=False),
    st_embedder=dict(
        type='BaseEmbedder',
        in_dims=2, 
        nb_freqs=10, 
        scale=1,
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
        use_dirs=True),
    nelf_field=dict(
        type='NeLFEPIField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=2*2*(10),
        color_dims=8,
        color_uv_dims=8,
        use_sin=False),
    render_params=dict( # default render cfg; train cfg
        n_samples=64,
        perturb=True,
        alpha_noise_std=1.0,
        inv_depth=False,
        use_dirs=True,
        background='white',
        max_rays_num=1024,),)

# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='AugLLFFDataset',
            datadir='~/data/3d/nerf/nerf_llff_data/fern', 
            factor=8, 
            # batch_size=1024*16,
            batch_size=1024*4,
            # batch_size=-1,
            split='train', 
            spherify=False, 
            no_ndc=True, 
            holdout=8,
            to_cuda=True),
        times=100),
    val=dict(
        type='LLFFDataset',
        datadir='~/data/3d/nerf/nerf_llff_data/fern', 
        factor=8, 
        batch_size=-1,
        split='val', 
        spherify=False, 
        no_ndc=True, 
        holdout=8),)

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='Exp', gamma=0.1**(1/50000), by_epoch=False) 
lr_config = dict(policy='Step', step=[40,80,120,160,180], gamma=0.5, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=200)
# misc settings
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', interval_exp_name=10000),
        # dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(
    interval=1,
    render_params=dict(
        n_samples=64,
        perturb=False,
        alpha_noise_std=0,
        inv_depth=False,
        use_dirs=True,
        background='white',
        max_rays_num=1024*3,))
extra_hooks = [dict(type='IterAdjustHook',), dict(type='LFPlaneUpdateModelByDataset',)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
