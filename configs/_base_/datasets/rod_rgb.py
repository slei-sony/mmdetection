_base_ = [
    "./coco_detection.py"
]

# dataset settings
dataset_type = 'RODDataset'
data_root = '/home/data/ROD_dataset/'

train_dataloader = dict(
    batch_size=10,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train-rgb-6000.json',
        data_prefix=dict(img='sRGB/'),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=_base_.train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val-rgb-1600.json',
        data_prefix=dict(img='sRGB/'),
        test_mode=True,
        pipeline=_base_.test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val-rgb-1600.json',
        data_prefix=dict(img='sRGB/'),
        test_mode=True,
        pipeline=_base_.test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val-rgb-1600.json',
    metric='bbox',
    format_only=False, classwise=True)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val-rgb-1600.json',
    metric='bbox',
    format_only=False, classwise=True)

# custom_hooks = [
#     dict(type="VariableLoggerHook"),
#     dict(type='TorchinfoHook', input_size=(4, 620, 400))
# ]
