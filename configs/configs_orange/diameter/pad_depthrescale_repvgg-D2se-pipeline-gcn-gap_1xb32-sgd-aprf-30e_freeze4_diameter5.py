_base_ = [
    '../_base_/models/repvgg-B3_lbs-mixup_in1k.py',
    #'../_base_/datasets/imagenet_bs64_pil_resize.py',
    #'../_base_/schedules/imagenet_bs256_200e_coslr_warmup.py',

    #'../_base_/datasets/orange21_bs64.py',
    #'../_base_/datasets/orange_grade_bs64.py',
    #'../_base_/datasets/orange_diameter_bs64.py',
 
    '../_base_/schedules/imagenet_bs256.py',
    #'../_base_/schedules/imagenet_bs2048_AdamW.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        arch='D2se',
        frozen_stages=4,
        init_cfg=dict(
            #_delete_=True, 
            type='Pretrained',
            #checkpoint='checkpoints/mmcls/repvgg-D2se_3rdparty_4xb64-autoaug-lbs-mixup-coslr-200e_in1k_20210909-cf3139b7.pth',
            checkpoint='mmgzb/work_dirs/pad_depthrescale_repvgg-D2se-pipeline_in1k-pre_1xb32-sgd-aprf-30e_diameter5/epoch_29.pth',
            prefix='backbone',
            )
    ),
    neck = dict(
        type='UnitGCNGAP'
    ),
    head = dict(
        num_classes=5,
        in_channels=512,
        loss=dict(
            num_classes=5),
    ),
    train_cfg=dict(
        augments=dict(type='BatchMixup', alpha=0.2, num_classes=5,
                      prob=1.))
)


# dataset setting
dataset_type = 'OrangeDiameter'
images_dir = 'data/orange/colorIMG_6view'
ann_file_dir = 'data/orange/statics/ann_diameter5'
label_type = 'name2diameter'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True,)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='RandomResizedCrop', size=224),
    #dict(type='RandomCrop', size=32, padding=4),
    dict(type='BboxCrop'),
    dict(type='ImageRescale'),
    dict(type='Pad', size=(400, 420)),
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
    dict(type='BboxCrop'),
    dict(type='ImageRescale'),
    dict(type='Pad', size=(400, 420)),
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=32, # for single GPU 8xb32
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_prefix=images_dir,
        ann_file='{}/{}_6view_train_bbox.txt'.format(ann_file_dir, label_type),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=images_dir,
        ann_file='{}/{}_6view_val_bbox.txt'.format(ann_file_dir, label_type),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix=images_dir,
        ann_file='{}/{}_6view_test_bbox.txt'.format(ann_file_dir, label_type),
        pipeline=test_pipeline))
#evaluation = dict(interval=1, metric='accuracy')
evaluation = dict(interval=1, metric=["accuracy", "precision", "recall", "f1_score"], metric_options=dict(topk=(1, 2)))


optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
#optimizer = dict(type='AdamW', lr=0.0015, weight_decay=0.3)
#optimizer_config = dict(grad_clip=dict(max_norm=1.0))

# learning policy
lr_config = dict(policy='step', step=[15, 20, 25])
runner = dict(type='EpochBasedRunner', max_epochs=30)
log_config = dict(interval=20)

# runtime setting
work_dir = './mmgzb/work_dirs/pad_depthrescale_repvgg-D2se-pipeline-gcn-gap_1xb32-sgd-aprf-30e_freeze4_diameter5'


