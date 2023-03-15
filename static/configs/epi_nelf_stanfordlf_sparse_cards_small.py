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
        std=64,
        include_input=False),
    # st_embedder=dict(
    #     type='BaseEmbedder',
    #     in_dims=2, 
    #     nb_freqs=10, 
    #     include_input=False),
    epi_embedder=dict(
        type='NormalEmbedder',
        in_dims=1, 
        nb_freqs=5,
        std=8,
        include_input=False),
    nelf_field=dict(
        type='NeLFEPIField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=2*2*(10),
        color_uv_dims=8,
        use_sin=False),
    epi_field=dict(
        type='EPIField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=2*2*(10),
        epi_emb_dims=2*5, 
        color_uv_dims=8,
        use_sin=False),
    epi_sample_num=128,
    pixel_move=[-50, 50], 
    near=0.1343,
    far=1.0894,
    epi_converge_iter=250*80, 
    epi_converge_range=0.5,
    # epi_converge_range=1,
    epi_smooth_weight=5e-5,
    render_params=dict(aug_points=512, perturb=True))

# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='BatchStanfordLFDataset',
            base_dir='~/data/3d/StanfordLF/cards-small/rectified', 
            # downsample=1,
            downsample=2,
            scale=1024,
            batch_size=int(1024),
            # keep_idx=['_00_00_', '_00_16_', '_16_00_', '_16_16_'],
            keep_idx=['_04_04_', '_04_12_', '_12_04_', '_12_12_'],
            # keep_idx=['_00_00_', '_00_16_', '_16_00_', '_16_16_'],
            # keep_idx=['_00_00_', '_00_08_', '_00_16_', 
            #           '_08_00_', '_08_08_', '_08_16_', 
            #           '_16_00_', '_16_08_', '_16_16_'],
            perturb=True,
            split='train',),
        times=100),
    val=dict(
        type='BatchStanfordLFDataset',
        base_dir='~/data/3d/StanfordLF/cards-small/rectified', 
        # downsample=1,
        downsample=2,
        scale=1024,
        batch_size=-1,
        keep_idx=['_08_06_', '_08_10_', '_06_08_', '_10_08_',],
        # keep_idx=['_04_04_', '_04_10_', '_10_04_', '_10_10_',],
        split='val',))

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Step', step=[50,100,150], gamma=0.5, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=200)
# misc settings
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=100,
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
