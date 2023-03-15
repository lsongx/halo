# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='NeLFEPI',
    embedder=dict(
        type='NormalEmbedder',
        in_dims=2, 
        nb_freqs=16, 
        std=128,
        include_input=True),
    # out_dim = (2*in_dims*nb_freqs + in_dims) if include_input else (2*in_dims*nb_freqs)
    field=dict(
        type='NeLFEPIField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=66, # 2*2*16+2
        color_dims=32,
        color_uv_dims=12,
        use_sin=True),
    epi_reg_weight=2,
    render_params=dict(aug_points=-1))

# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='StanfordLFDataset',
            # base_dir='~/data/3d/StandfordLF/cards-fine',
            # base_dir='~/data/3d/StandfordLF/cards-coarse', 
            # base_dir='~/data/3d/StandfordLF/lego-truck',
            base_dir='~/data/3d/StandfordLF/lego-knights', 
            downsample=2,
            # downsample=4,
            scale=1024,
            batch_size=-1,
            testskip=8,
            split='train',
            add_aug=False),
        times=100),
        # times=1),
    val=dict(
        type='StanfordLFDataset',
        # base_dir='~/data/3d/StandfordLF/cards-fine',
        # base_dir='~/data/3d/StandfordLF/cards-coarse', 
        # base_dir='~/data/3d/StandfordLF/lego-truck',
        base_dir='~/data/3d/StandfordLF/lego-knights', 
        downsample=2,
        # downsample=4,
        scale=1024,
        batch_size=-1,
        testskip=8,
        split='val',
        add_aug=False))

# optimizer
# optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer = dict(type='AdamW', lr=5e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='Exp', gamma=0.1**(1/75000), by_epoch=False) 
lr_config = dict(policy='Poly', min_lr=5e-5, by_epoch=False) 
runner = dict(type='EpochBasedRunner', max_epochs=200)
# misc settings
checkpoint_config = dict(interval=1, max_keep_ckpts=30)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', interval_exp_name=10000),
        # dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(
    interval=1,
    render_params=dict(batch_ray_forward=1024))
extra_hooks = [dict(type='IterAdjustHook',), dict(type='EPIUpdateModelByDataset')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
