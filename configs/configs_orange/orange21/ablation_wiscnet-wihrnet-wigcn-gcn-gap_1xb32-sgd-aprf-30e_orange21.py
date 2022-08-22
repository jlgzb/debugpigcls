_base_ = [
    '../_base_/models/hrnet/hrnet-w48.py',
    #'../_base_/datasets/imagenet_bs64_pil_resize.py',
    #'../_base_/schedules/imagenet_bs256_200e_coslr_warmup.py',
    '../_base_/datasets/orange21_bs64.py',   
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]

# model
# model setting
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/mmcls/hrnet-w48_3rdparty_8xb32-ssld_in1k_20220120-d0459c38.pth',
            prefix='backbone',
            )
        ),
    neck=[
        dict(type='HRFuseScales', in_channels=(48, 96, 192, 384)),
        dict(type='UnitGCNGAPHrn'),
    ],
    head=dict(
        num_classes=21,
        in_channels=512,
    )
)

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
work_dir = './mmgzb/work_dirs/ablation_wiscnet-wihrnet-wigcn-gcn-gap_1xb32-sgd-aprf-30e_orange21'


