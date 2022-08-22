_base_ = [
    '../_base_/models/twins_svt_base.py',
    #'../_base_/datasets/imagenet_bs64_swin_224.py',
    #'../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    #'../_base_/datasets/orange21_bs64.py',
    '../_base_/datasets/orange_grade_bs64.py',
    #'../_base_/datasets/orange_diameter_bs64.py',  
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]

# model setting
model = dict(
    backbone=dict(
        init_cfg=dict(
            #_delete_=True,
            type='Pretrained',
            checkpoint='checkpoints/mmcls/twins-svt-base_3rdparty_8xb128_in1k_20220126-e31cc8e9.pth',
            prefix='backbone',
            )
        ),
    head=dict(num_classes=4),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=4, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=4, prob=0.5)
    ])
)


data = dict(samples_per_gpu=64)
evaluation = dict(metric_options=dict(topk=(1, 2)))

# train setting
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(policy='step', step=[15, 20, 25])
runner = dict(type='EpochBasedRunner', max_epochs=30)

log_config = dict(interval=20)

# runtime setting
work_dir = './mmgzb/work_dirs/twins-svt-base_in1k-pre_1xb64-sgd-aprf-30e_grade4'

