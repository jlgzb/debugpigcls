_base_ = [
    '../_base_/models/conformer/base-p16.py',
    '../_base_/datasets/orange21_bs64.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]

# model setting
model = dict(
    backbone=dict(
        
        init_cfg=dict(
            _delete_=True,
            type='Pretrained',
            checkpoint='checkpoints/mmcls/conformer-base-p16_3rdparty_8xb128_in1k_20211206-bfdf8637.pth',
            prefix='backbone',
            )
    ),
    head=dict(
        num_classes=21,
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=21, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=21, prob=0.5)
    ])
)

data = dict(
    samples_per_gpu=32,
)


# train setting
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(policy='step', step=[15, 20, 25])
runner = dict(type='EpochBasedRunner', max_epochs=30)

log_config = dict(interval=20)


# runtime setting
work_dir = './mmgzb/work_dirs/conformer-base-p16_in1k-pre_1xb32-sgd-aprf-30e_orange21'