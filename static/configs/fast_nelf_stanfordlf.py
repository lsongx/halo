# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='FastNeLF',
    st_embedder=dict(
        type='NormalEmbedder',
        in_dims=2, 
        nb_freqs=64, 
        std=16,
        include_input=True),
    uv_embedder=dict(
        type='NormalEmbedder',
        in_dims=2, 
        # nb_freqs=64, 
        nb_freqs=16, 
        # std=256,
        std=16,
        include_input=True),
    # out_dim = (2*in_dims*nb_freqs + in_dims) if include_input else (2*in_dims*nb_freqs)
    st_basis_field=dict(
        type='NeLFBasisField',
        nb_layers=6, 
        hid_dims=128, 
        emb_dims=258, # 2*2*16+2
        # emb_dims=66, # 2*2*16+2
        out_dims=64,
        use_sin=True),
    uv_st_field=dict(
        type='NeLFWeightField',
        nb_layers=6, 
        hid_dims=256, 
        st_emb_dims=258, # 2*2*64+2
        # st_emb_dims=66, # 2*2*16+2
        # uv_emb_dims=258, # 2*2*64+2
        uv_emb_dims=66,
        out_dims=64,
        use_sin=True),
    render_params=dict())

# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='StanfordLFDataset',
            base_dir='~/data/3d/StandfordLF/cards-fine',
            # base_dir='~/data/3d/StandfordLF/cards-coarse', 
            # base_dir='~/data/3d/StandfordLF/lego-truck',
            # base_dir='~/data/3d/StandfordLF/lego-knights', 
            downsample=2,
            # downsample=4,
            scale=1024,
            batch_size=-1,
            testskip=8,
            split='train',
            add_aug=False),
        times=300),
    val=dict(
        type='StanfordLFDataset',
        base_dir='~/data/3d/StandfordLF/cards-fine',
        # base_dir='~/data/3d/StandfordLF/cards-coarse', 
        # base_dir='~/data/3d/StandfordLF/lego-truck',
        # base_dir='~/data/3d/StandfordLF/lego-knights', 
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
    render_params=dict())
extra_hooks = [dict(type='IterAdjustHook',),]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
