_base_ = [
    '../../../configs/_base_/models/isanet_r50-d8.py', '../../../configs/_base_/datasets/manipulation.py',
    '../../../configs/_base_/default_runtime.py', '../../../configs/_base_/schedules/schedule_80k.py'
]

model = dict(
    decode_head=dict(num_classes=2,
                    # out_channels=2,
                    # loss_decode=dict(
                    #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                    ),
    auxiliary_head=dict(num_classes=2,
                    # out_channels=2,
                    # loss_decode=dict(
                    #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
                    )) #自定义篡改数据集为只有一类，加背景 num_class=2

# dataset settings
dataset_type = 'ManipulationDataset' #change
data_root = '../../data/tianchi/manipulation/manipulation_like_ade/'  

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (320, 320)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 640), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2560, 640),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))