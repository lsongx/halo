# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='NeLFEPIOcc360',
    embedder=dict(
        type='NormalEmbedder',
        in_dims=2, 
        nb_freqs=128, 
        std=32,
        include_input=False),
    epi_embedder=dict(
        type='NormalEmbedder',
        in_dims=1, 
        nb_freqs=8, 
        std=16,
        include_input=True),
    hf_color_embedder=None,
    field=dict(
        type='NeLFEPIField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=2*2*(128),
        color_dims=8,
        color_uv_dims=8,
        positive_color_code=True,
        hf_color_subnet_dim=0,
        use_sin=True),
    occ_field=dict(
        type='NeLFEPIOccField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=2*2*(128),
        epi_emb_dims=17, 
        color_dims=8,
        positive_color_code=True,
        use_sin=True),
    epi_reg_weight=2,
    epi_sample_num=128,
    occ_consist_weight=0,
    pixel_move=40,
    near=0,
    far=3.2,
    epi_converge_iter=2200*100, 
    epi_converge_range=0.2,
    render_params=dict(aug_points=512, perturb=True))

# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='SyntheticSphereCoordDataset',
            base_dir='~/data/3d/nerf/nerf_synthetic/lego', 
            half_res=True,
            batch_size=1024,
            background='white',
            precrop_frac=0.5,
            testskip=8,
            rad0=12,
            rad1=16.25,
            uv_precision=4,
            split='train'),
        times=100),
    val=dict(
        type='SyntheticSphereCoordDataset',
        base_dir='~/data/3d/nerf/nerf_synthetic/lego', 
        half_res=True,
        batch_size=-1,
        background='white',
        precrop_frac=0.5,
        testskip=8,
        rad0=9,
        rad1=16.25,
        uv_precision=4,
        split='val'))

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Step', step=[50,100,150], gamma=0.5, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=200)
# misc settings
checkpoint_config = dict(interval=1, max_keep_ckpts=30)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', interval_exp_name=10000),
    ])
evaluation = dict(
    interval=1,
    extra_log='epi_map',
    render_params=dict(batch_ray_forward=1024, perturb=False))
extra_hooks = [dict(type='IterAdjustHook',), dict(type='SphereUpdateModelByDataset',)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
