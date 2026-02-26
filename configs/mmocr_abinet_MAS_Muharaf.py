_base_ = [
    '../_base_/datasets/mjsynth.py',
    '../_base_/datasets/synthtext.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_base.py',
    '_base_abinet.py',
]

load_from = 'path/to/data/mmocr/checkpoints/abinet_pretrain.pth'

# 1. Dataset Settings
dataset_type = 'RecogTextDataset'
data_root = 'path/to/data/PaddleOCR/datasets/mmocr_anno'

train_ann_file = 'train_mmocr.jsonl'
val_ann_file = 'val_mmocr.jsonl'
dict_file = 'path/to/data/PaddleOCR/datasets/mmocr_anno/dict_mmocr.txt'

train_pipeline = [
    dict(type='LoadImageFromFile', ignore_empty=True, min_size=2),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=(640, 64)), # 640x64
    dict(
        type='RandomApply',
        prob=0.5,
        transforms=[
            dict(
                type='RandomChoice',
                transforms=[
                    dict(
                        type='RandomRotate',
                        max_angle=15,
                    ),
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomAffine',
                        degrees=15,
                        translate=(0.3, 0.3),
                        scale=(0.5, 2.),
                        shear=(-45, 45),
                    ),
                    dict(
                        type='TorchVisionWrapper',
                        op='RandomPerspective',
                        distortion_scale=0.5,
                        p=1,
                    ),
                ])
        ],
    ),
    dict(
        type='RandomApply',
        prob=0.25,
        transforms=[
            dict(type='PyramidRescale'),
            # Removed Albu due to potential version mismatch
            # dict(
            #     type='mmdet.Albu',
            #     transforms=[
            #         dict(type='GaussNoise', var_limit=(20, 20), p=0.5),
            #         dict(type='MotionBlur', blur_limit=7, p=0.5),
            #     ]),
        ]),
    dict(
        type='RandomApply',
        prob=0.25,
        transforms=[
            dict(
                type='TorchVisionWrapper',
                op='ColorJitter',
                brightness=0.5,
                saturation=0.5,
                contrast=0.5,
                hue=0.1),
        ]),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 64)),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

# 2. Model Settings
# Override dictionary path in model
dictionary = dict(
    type='Dictionary',
    dict_file=dict_file,
    with_start=True,
    with_end=True,
    same_start_end=True,
    with_padding=False,
    with_unknown=True)

model = dict(
    type='ABINet',
    backbone=dict(type='ResNetABI'),
    encoder=dict(
        type='ABIEncoder',
        n_layers=3,
        n_head=8,
        d_model=512,
        d_inner=2048,
        dropout=0.1,
        max_len=3000, # 16 * 160 = 2560 features
    ),
    decoder=dict(
        type='ABIFuser',
        vision_decoder=dict(
            type='ABIVisionDecoder',
            in_channels=512,
            num_channels=64,
            attn_height=16, # 64/4
            attn_width=160, # 640/4
            attn_mode='nearest',
            init_cfg=dict(type='Xavier', layer='Conv2d')),
        module_loss=dict(type='ABIModuleLoss', letter_case='lower'),
        postprocessor=dict(type='AttentionPostprocessor'),
        dictionary=dictionary,
        max_seq_len=192, 
    ),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))

# 3. Dataloader Settings
arabic_train = dict(
    type='RecogTextDataset',
    data_root=data_root,
    ann_file=train_ann_file,
    pipeline=train_pipeline)

arabic_test = dict(
    type='RecogTextDataset',
    data_root=data_root,
    ann_file=val_ann_file,
    pipeline=test_pipeline)

train_dataloader = dict(
    batch_size=24, # Reduced from 64 to avoid OOM with 640x64 images
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=arabic_train)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=arabic_test)

val_dataloader = test_dataloader

# 4. Evaluation and Schedule
val_evaluator = dict(
    dataset_prefixes=['Arabic'],
    metrics=[
        dict(
            type='WordMetric',
            mode=['exact', 'ignore_case', 'ignore_case_symbol'],
            valid_symbol='[^A-Z^a-z^0-9^\\u0600-\\u06FF]'), # Added 0600-06FF for Arabic
        dict(type='CharMetric', valid_symbol='[^A-Z^a-z^0-9^\\u0600-\\u06FF]'),
        dict(type='OneMinusNEDMetric', valid_symbol='[^A-Z^a-z^0-9^\\u0600-\\u06FF]'),
    ])
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(lr=1e-4))
train_cfg = dict(max_epochs=100, val_interval=1)

# Model saving
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5, save_best='Arabic/recog/word_acc_ignore_case', rule='greater'),
    logger=dict(type='LoggerHook', interval=50),
)

auto_scale_lr = dict(base_batch_size=64 * 1)
