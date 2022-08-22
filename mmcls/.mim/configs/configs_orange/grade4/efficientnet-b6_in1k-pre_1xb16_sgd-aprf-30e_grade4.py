_base_ = [
    '../_base_/models/efficientnet_b6.py',
    '../_base_/datasets/orange_grade_bs64.py',   
    '../_base_/schedules/imagenet_bs256.py',  
    '../_base_/default_runtime.py'          
]

model = dict(
    backbone=dict(
        type='EfficientNet', 
        arch='b6',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/mmcls/efficientnet-b6_3rdparty_8xb32-aa_in1k_20220119-45b03310.pth',
            prefix='backbone',
        ),
    ),
    head=dict(
        num_classes=4,
        topk=(1, 2),
    )
)

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True,)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='RandomResizedCrop', size=224),
    #dict(type='RandomCrop', size=32, padding=4),
    dict(type='BboxCrop'),
    dict(type='Resize', size=528),
    dict(
        type='RandomResizedCrop',
        size=528,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='RandomResizedCrop', size=224),
    dict(type='BboxCrop'),
    dict(type='Resize', size=528),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=16,
)

evaluation = dict(
    metric_options=dict(topk=(1, 2))
)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(policy='step', step=[15, 20, 25])
runner = dict(type='EpochBasedRunner', max_epochs=30)

log_config = dict(interval=50)

# runtime setting
work_dir = './mmgzb/work_dirs/efficientnet-b6_in1k-pre_1xb16-sgd-aprf-30e_grade4'