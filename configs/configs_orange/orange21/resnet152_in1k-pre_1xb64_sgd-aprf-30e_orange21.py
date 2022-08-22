_base_ = [
    '../_base_/models/resnet152.py',     
    '../_base_/datasets/orange21_bs64.py',   
    '../_base_/schedules/imagenet_bs256.py',  
    '../_base_/default_runtime.py'          
]

# model setting
model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/mmcls/resnet152_8xb32_in1k_20210901-4d7582fa.pth',
            prefix='backbone',
            )
        ),
    head=dict(num_classes=21)
)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(policy='step', step=[15, 20, 25])
runner = dict(type='EpochBasedRunner', max_epochs=30)
log_config = dict(interval=20)

# runtime setting
work_dir = './mmgzb/work_dirs/resnet152_in1k-pre_1xb64-sgd-aprf-30e_orange21'