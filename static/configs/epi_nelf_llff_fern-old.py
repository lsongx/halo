# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='EPINeLF',
    # uv_embedder=dict(
    #     type='NormalEmbedder',
    #     in_dims=2,
    #     nb_freqs=16,
    #     std=128,
    #     scale=1,
    #     include_input=False),
    # st_embedder=dict(
    #     type='NormalEmbedder',
    #     in_dims=2,
    #     nb_freqs=16,
    #     std=128,
    #     scale=1,
    #     include_input=False),
    # epi_embedder=dict(
    #     type='NormalEmbedder',
    #     in_dims=2,
    #     nb_freqs=16,
    #     std=128,
    #     scale=1,
    #     include_input=False),
    uv_embedder=dict(
        type='BaseEmbedder',
        in_dims=2, 
        nb_freqs=10, 
        include_input=False),
    st_embedder=dict(
        type='BaseEmbedder',
        in_dims=2, 
        nb_freqs=10, 
        include_input=False),
    epi_embedder=dict(
        type='BaseEmbedder',
        in_dims=1, 
        nb_freqs=10, 
        include_input=False),
    epi_field=dict(
        type='NeLFEPIOccField',
        nb_layers=8,
        hid_dims=256,
        emb_dims=2*2*10,
        epi_emb_dims=2*10,
        color_dims=8,
        positive_color_code=True,
        use_sin=False),
    nelf_field=dict(
        type='NeLFEPIField',
        nb_layers=8,
        hid_dims=256,
        emb_dims=2*2*10,
        color_dims=8,
        color_uv_dims=6,
        use_sin=False),
    epi_sample_num=128,
    epi_near=0.05,
    epi_far=1.,
    epi_converge_iter=2200*100,
    # epi_converge_range=0.2,
    epi_converge_range=1,
    render_params=dict(aug_points=512, perturb=True))

# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='AugTwoLLFFDataset',
            datadir='~/data/3d/nerf/nerf_llff_data/fern', 
            factor=8, 
            # batch_size=1024*16,
            batch_size=1024,
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
lr_config = dict(policy='Poly', min_lr=5e-5, by_epoch=False) 
runner = dict(type='EpochBasedRunner', max_epochs=200)
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
    render_params=dict(max_rays_num=1024))
extra_hooks = [dict(type='IterAdjustHook',), dict(type='LFPlaneUpdateModelByDataset',)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
