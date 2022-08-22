_base_ = [
    '../_base_/models/vgg19bn.py',
    #'../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/datasets/orange_diameter_bs64.py',   
    '../_base_/schedules/imagenet_bs256.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='VGG', depth=19, norm_cfg=dict(type='BN'), num_classes=5,
        init_cfg=dict(
            _delete_=True,
            type='Pretrained',
            checkpoint='checkpoints/mmcls/vgg19_bn_batch256_imagenet_20210208-da620c4f.pth',
            prefix='backbone',
            ),
    )
)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(policy='step', step=[15, 20, 25])
runner = dict(type='EpochBasedRunner', max_epochs=30)

log_config = dict(interval=20)

# runtime setting
work_dir = './mmgzb/work_dirs/vgg19bn_in1k-pre_1xb64-sgd-aprf_30e_diameter'