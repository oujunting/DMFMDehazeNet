# model settings

model = dict(
    type='Pix2PixHD',
    render=True,
    generator=dict(
        type='MixDehazeNet',
        att=True
    ),
    ssim_loss=dict(type='SSIMLoss', loss_weight=1.0),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(direction='a2b')  # model default: a2b
test_cfg = dict(direction='a2b')

# dataset settings
train_dataset_type = 'BlemishDataset'
val_dataset_type = 'BlemishDataset'
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(
        type='LoadBlemishPairedImage',
        io_backend='disk',
        key=['A', 'B'],
        flag='color'),
    dict(
        type='Resize',
        keys=['img_a', 'img_b'],
        scale=(600, 600),
        interpolation='bicubic'),
    dict(type='FixedCrop', keys=['img_a', 'img_b'], crop_size=(512, 512)),
    dict(type='Flip', keys=['img_a', 'img_b'], direction='horizontal'),
    dict(type='RescaleToZeroOne', keys=['img_a', 'img_b']),
    dict(
        type='Normalize', keys=['img_a', 'img_b'], to_rgb=True,
        **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img_a', 'img_b']),
    dict(
        type='Collect',
        keys=['img_a', 'img_b'],
        meta_keys=['img_a_path', 'img_b_path'])
]
test_pipeline = [
    dict(
        type='LoadBlemishPairedImage',
        io_backend='disk',
        key=['A', 'B'],
        flag='color'),
    dict(
        type='Resize',
        keys=['img_a', 'img_b'],
        scale=(512, 512),
        interpolation='bicubic'),
    dict(type='RescaleToZeroOne', keys=['img_a', 'img_b']),
    dict(
        type='Normalize', keys=['img_a', 'img_b'], to_rgb=True,
        **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img_a', 'img_b']),
    dict(
        type='Collect',
        keys=['img_a', 'img_b'],
        meta_keys=['img_a_path', 'img_b_path'])
]
data_root = '/home/featurize/data/RESIDE-IN/'
data = dict(
    #batch size
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=train_dataset_type,
        dataroot=[
                  data_root + "train"
        ],
        root_name=dict(
            A='hazy',
            B='GT'
        ),
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=val_dataset_type,
        dataroot=[
            data_root + "test"
        ],
        root_name=dict(
            A='hazy',
            B='GT'
        ),
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=val_dataset_type,
        dataroot=[
            data_root + "test"
        ],
        root_name=dict(
            A='hazy',
            B='GT'
        ),
        pipeline=test_pipeline,
        test_mode=True))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)))

# learning policy
lr_config = dict(policy='Fixed', by_epoch=False)

# checkpoint saving
checkpoint_config = dict(interval=2500, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=2500, save_image=False)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
visual_config = None

# runtime settings
#epoch
total_iters = 10000
cudnn_benchmark = False
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = f'./work_dirs/{{ fileBasenameNoExtension }}'

