_base_ = '../../third_party/mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'

data_root = 'data/'
class_names = (
    'Building',
    'Fully_loaded_cargo_ship',
    'Fully_loaded_container_ship',
    'Lock_gate',
    'Tree',
    'Unladen_cargo_ship',
    'Unladen_container_ship',
    'Fully_loaded_cargo_fleet',
    'Unladen_cargo_fleet',
    'Lock_footbridge',
    'Crew_member',
    'Mooring_line',
    'Tugboat',
    'Unknown_vessel',
)
metainfo = dict(classes=class_names)

model = dict(bbox_head=dict(num_classes=len(class_names)))

train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(960, 960),
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow', backend_args=None),
    dict(type='Resize', scale=(960, 960), keep_ratio=True),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='2d_annotations/instances_train.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='2d_annotations/instances_val.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        pipeline=test_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='2d_annotations/instances_test.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '2d_annotations/instances_val.json',
    metric='bbox')
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '2d_annotations/instances_test.json',
    metric='bbox')

max_epochs = 20
stage2_num_epochs = 2
base_lr = 0.002
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
]

default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50))

optim_wrapper = dict(optimizer=dict(lr=0.001))

work_dir = 'outputs/mmdet2d/navlock_rtmdet_s'
