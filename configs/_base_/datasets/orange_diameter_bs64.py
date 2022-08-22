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
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=64, # for single GPU 8xb32
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
evaluation = dict(interval=1, metric=["accuracy", "precision", "recall", "f1_score"])
