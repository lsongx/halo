# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='EPINeLFUvSt',
    uv_embedder=dict(
        type='NormalEmbedder',
        in_dims=2, 
        nb_freqs=10, 
        std=16,
        include_input=False),
    st_embedder=dict(
        type='NormalEmbedder',
        in_dims=2, 
        nb_freqs=10, 
        std=16,
        include_input=False),
    # st_embedder=dict(
    #     type='BaseEmbedder',
    #     in_dims=2, 
    #     nb_freqs=10, 
    #     include_input=False),
    epi_embedder=dict(
        type='NormalEmbedder',
        in_dims=1, 
        nb_freqs=10,
        std=16,
        include_input=False),
    nelf_field=dict(
        type='NeLFEPIField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=2*2*(10),
        color_uv_dims=8,
        # use_sin=True),
        use_sin=False),
    epi_field=dict(
        type='EPIField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=2*2*(10),
        epi_emb_dims=2*10, 
        # color_uv_dims=8,
        color_uv_dims=0,
        # use_sin=True),
        use_sin=False),
    epi_sample_num=128,
    # pixel_move=-1.0,
    # pixel_move=[-150, 80], 
    pixel_move=[-180, 180], 
    near=0.1343,
    far=1.0894,
    epi_converge_iter=250*50, 
    epi_converge_range=0.2,
    # epi_converge_range=0.5,
    # epi_smooth_weight=0,
    epi_smooth_weight=1e-6,
    render_params=dict(aug_points=512, perturb=True))
    # render_params=dict(aug_points=512, perturb=False))

# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='BatchStanfordLFDataset',
            # base_dir='~/data/3d/StanfordLF/cards-small/rectified',
            # base_dir='~/data/3d/StanfordLF/cards-coarse/rectified', 
            # base_dir='~/data/3d/StanfordLF/lego-truck/rectified',
            base_dir='~/data/3d/StanfordLF/lego-knights/rectified', 
            # base_dir='~/data/3d/StanfordLF/eucalyptus-flowers/rectified', 
            # downsample=1,
            downsample=2,
            # downsample=4,
            scale=1024,
            batch_size=int(1024*3),
            # batch_size=int(1024),
            keep_idx=['_00_00_', '_00_16_', '_08_08_', '_16_00_', '_16_16_'],
            # keep_idx=['_00_00_', '_00_16_', '_16_00_', '_16_16_'],
            perturb=True,
            split='train',),
        times=50),
    val=dict(
        type='BatchStanfordLFDataset',
        # base_dir='~/data/3d/StanfordLF/cards-small/rectified',
        # base_dir='~/data/3d/StanfordLF/cards-coarse/rectified', 
        # base_dir='~/data/3d/StanfordLF/lego-truck/rectified',
        base_dir='~/data/3d/StanfordLF/lego-knights/rectified', 
        # base_dir='~/data/3d/StanfordLF/eucalyptus-flowers/rectified', 
        # downsample=1,
        downsample=2,
        # downsample=4,
        scale=1024,
        batch_size=-1,
        keep_idx=['_04_08_', '_10_08_', '_08_04_', '_08_10_',],
        split='val',))

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
# optimizer = dict(
#     # type='AdamW', lr=5e-4, 
#     type='Adam', lr=5e-4, betas=(0.9, 0.999),
#     paramwise_cfg=dict(custom_keys={'.field.color_code_hf_subnet': dict(lr_mult=0.1)}))
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='Exp', gamma=0.1**(1/75000), by_epoch=False)
# lr_config = dict(policy='Poly', min_lr=5e-6, by_epoch=False)
lr_config = dict(policy='Step', step=[50,100,150], gamma=0.5, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=200)
# lr_config = dict(policy='Step', step=[150,200,250], gamma=0.5, by_epoch=True)
# runner = dict(type='EpochBasedRunner', max_epochs=300)
# misc settings
checkpoint_config = dict(interval=1, max_keep_ckpts=30)
log_config = dict(
    interval=50,
    # interval=1,
    hooks=[
        dict(type='TextLoggerHook', interval_exp_name=10000),
        # dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(
    interval=1,
    extra_log='epi_map',
    render_params=dict(max_rays_num=1024, perturb=False))
extra_hooks = [
    dict(type='IterAdjustHook',), 
    dict(type='EPIUpdateModelByDataset', grid_num=1)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
