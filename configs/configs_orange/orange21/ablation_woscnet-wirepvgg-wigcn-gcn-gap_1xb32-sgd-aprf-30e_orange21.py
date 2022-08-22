_base_ = [
    '../_base_/models/repvgg-B3_lbs-mixup_in1k.py',
    #'../_base_/datasets/imagenet_bs64_pil_resize.py',
    #'../_base_/schedules/imagenet_bs256_200e_coslr_warmup.py',
    '../_base_/datasets/orange21_bs64.py',   
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]

# model
model = dict(
    backbone=dict(
        arch='D2se',
        frozen_stages=2,
        init_cfg=dict(
            #_delete_=True, 
            type='Pretrained',
            checkpoint='checkpoints/mmcls/repvgg-D2se_3rdparty_4xb64-autoaug-lbs-mixup-coslr-200e_in1k_20210909-cf3139b7.pth',
            #checkpoint='mmgzb/work_dirs/repvgg-D2se_in1k-pre_1xb32-sgd-aprf-30e_orange21/epoch_26.pth',
            prefix='backbone',
            )
    ),
    neck = dict(
        type='UnitGCNGAP'
    ),
    head = dict(
        num_classes=21,
        in_channels=512,
        loss=dict(
            num_classes=21),
    ),
    train_cfg=dict(
        augments=dict(type='BatchMixup', alpha=0.2, num_classes=21,
                      prob=1.))
)

# data pipeline
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True,)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='RandomResizedCrop', size=224),
    #dict(type='RandomCrop', size=32, padding=4),
    #dict(type='BboxCrop'),
    dict(type='Resize', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='RandomResizedCrop', size=224),
    #dict(type='BboxCrop'),
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=32,
)

# schedule
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
evaluation = dict(
    metric_options=dict(topk=(1, 2))
)

# learning policy
lr_config = dict(policy='step', step=[15, 20, 25])
runner = dict(type='EpochBasedRunner', max_epochs=30)
log_config = dict(interval=20)

# runtime setting
work_dir = './mmgzb/work_dirs/ablation_woscnet-wirepvgg-wigcn-gcn-gap_1xb32-sgd-aprf-30e_orange21'


