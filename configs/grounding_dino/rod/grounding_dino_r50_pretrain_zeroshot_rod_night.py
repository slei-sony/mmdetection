_base_ = '../grounding_dino_r50_scratch_8xb2_1x_coco.py'

dataset_type = 'RODDataset'
data_root = '/home/data/ROD_dataset/'

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        ann_file='annotations/night-val-rgb-800.json',
        data_prefix=dict(img='sRGB/')))
test_dataloader = val_dataloader

# numpy < 1.24.0
val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root +
    'annotations/night-val-rgb-800.json',
    metric='bbox',
    format_only=False, classwise=True)
test_evaluator = val_evaluator
