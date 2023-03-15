# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='EPINeLFRays',
    uv_embedder=dict(
        type='NormalEmbedder',
        in_dims=2, 
        nb_freqs=10, 
        std=8,
        include_input=False),
        # include_input=True),
    lf_st_embedder=dict(
        type='NormalEmbedder',
        in_dims=2, 
        nb_freqs=10, 
        std=32,
        scale=8,
        include_input=False),
    # epi_embedder=dict(
    #     type='NormalEmbedder',
    #     in_dims=1, 
    #     nb_freqs=5,
    #     std=4,
    #     include_input=False),
    # uv_embedder=dict(
    #     type='BaseEmbedder',
    #     in_dims=2, 
    #     nb_freqs=10, 
    #     include_input=False),
    st_embedder=dict(
        type='BaseEmbedder',
        in_dims=2, 
        nb_freqs=10, 
        scale=8,
        include_input=False),
        # include_input=True),
    epi_embedder=dict(
        type='BaseEmbedder',
        in_dims=1, 
        nb_freqs=10, 
        scale=1,
        include_input=False),
        # include_input=True),
    nelf_field=dict(
        type='NeLFEPIField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=2*2*(10),#+2,
        # color_uv_dims=2*2*(10),#+2,
        color_uv_dims=0,#+2,
        use_sin=False),
    epi_field=dict(
        type='EPIField',
        nb_layers=8, 
        hid_dims=256, 
        emb_dims=2*2*(10),#+2,
        epi_emb_dims=2*10,#+1, 
        color_uv_dims=4,
        use_sin=False),
    epi_sample_num=128,
    # epi_converge_iter=1700*40, 
    epi_converge_iter=1200*10, 
    # epi_converge_range=1,
    epi_converge_range=0.1,
    epi_smooth_weight=0,
    stop_update=True,
    background='white',
    # epi_smooth_weight=1e-6,
    render_params=dict(aug_points=512, perturb=True))

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
            select_idx=[22,25,28,40,44,48],
            batching=True,
            to_cuda=True),
        times=8),
    val=dict(
        type='DTUDataset',
        datadir='~/data/3d/rs_dtu_4', 
        load_object='scan21',
        batch_size=-1,
        exclude_idx=[22,25,28,40,44,48],
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
    extra_log='epi_map',
    save_raw=True,
    render_params=dict(max_rays_num=1024, perturb=False))
extra_hooks = [dict(type='IterAdjustHook',), dict(type='LFPlaneUpdateModelByDataset',)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
