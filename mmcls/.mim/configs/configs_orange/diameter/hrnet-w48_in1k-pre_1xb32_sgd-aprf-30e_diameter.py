_base_ = [
    '../_base_/models/hrnet/hrnet-w48.py',
    #'../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/datasets/orange_diameter_bs64.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]

# model setting
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/mmcls/hrnet-w48_3rdparty_8xb32-ssld_in1k_20220120-d0459c38.pth',
            prefix='backbone',
            )
        ),
    head=dict(num_classes=5)
)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

data = dict(
    samples_per_gpu=32,
)

# learning policy
lr_config = dict(policy='step', step=[15, 20, 25])
runner = dict(type='EpochBasedRunner', max_epochs=30)
log_config = dict(interval=20)

# runtime setting
work_dir = './mmgzb/work_dirs/hrnet-w48_in1k-pre_1xb32-sgd-aprf-30e_diameter'
