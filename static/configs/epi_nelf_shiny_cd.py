# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='EPINeLFRays',
    uv_embedder=dict(
        type='NormalEmbedder',
        in_dims=2, 
        nb_freqs=10, 
        std=16,
        include_input=False),
    # st_embedder=dict(
    #     type='NormalEmbedder',
    #     in_dims=2, 
    #     nb_freqs=10, 
    #     std=64,
    #     scale=128,
    #     include_input=False),
    epi_embedder=dict(
        type='NormalEmbedder',
        in_dims=1, 
        nb_freqs=10,
        std=16,
        include_input=False),
    # uv_embedder=dict(
    #     type='BaseEmbedder',
    #     in_dims=2, 
    #     nb_freqs=10, 
    #     include_input=False),
    st_embedder=dict(
        type='BaseEmbedder',
        in_dims=2, 
        nb_freqs=10, 
        scale=256,
        include_input=False),
    # epi_embedder=dict(
    #     type='BaseEmbedder',
    #     in_dims=2, 
    #     nb_freqs=10, 
    #     scale=0.5,
    #     include_input=False),
    nelf_field=dict(
        type='NeLFEPIField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=2*2*(10),
        color_uv_dims=2*2*(10),
        use_sin=False),
    epi_field=dict(
        type='EPIField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=2*2*(10),
        epi_emb_dims=2*10, 
        color_uv_dims=2*2*(10),
        use_sin=False),
    epi_sample_num=128,
    epi_converge_iter=1000*100, 
    epi_converge_range=1,
    # epi_converge_range=0.5,
    epi_smooth_weight=1e-6,
    render_params=dict(aug_points=512, perturb=True))
    # render_params=dict(aug_points=512, perturb=False))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=5,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='AugShinyDataset',
            base_dir='~/data/3d/shiny/cd',
            llff_width=1008,
            batch_size=1024*3,
            split='train',
            batching=True,
            to_cuda=True,
            cache_size=512,
            holdout=8),
        times=50),
    val=dict(
        type='ShinyDataset',
        base_dir='~/data/3d/shiny/cd',
        llff_width=1008,
        batch_size=-1,
        split='val',
        batching=False,
        to_cuda=True,
        cache_size=512,
        holdout=8),)

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='Step', step=[20,40,60,80], gamma=0.5, by_epoch=True)
# runner = dict(type='EpochBasedRunner', max_epochs=100)
lr_config = dict(policy='Step', step=[30,60,90], gamma=0.5, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=150)
# misc settings
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook', interval_exp_name=10000),
        # dict(type='TensorboardLoggerHook', log_dir='./logs')
    ])
evaluation = dict(
    interval=1,
    extra_log='epi_map',
    render_params=dict(max_rays_num=1024, perturb=False))
extra_hooks = [dict(type='IterAdjustHook',), dict(type='LFPlaneUpdateModelByDataset',)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
