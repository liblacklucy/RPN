norm_cfg = dict(type='SyncBN', requires_grad=True)
img_size = 512
in_channels = 512
out_indices = [11]
model = dict(
    type='CrossPromptCLIP',
    pretrained='/data/ljh/pretrained/ViT-B-16.pt',
    context_length=77,
    backbone=dict(
        type='CPTCLIPVisionTransformer',
        layers=12,
        style='pytorch',
        patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.1,
        input_resolution=512,
        out_indices=[11],
        num_tokens=10,
        prompt_dim=768,
        total_d_layer=12),
    text_encoder=dict(
        type='CPTCLIPTextEncoder',
        context_length=77,
        style='pytorch',
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        out_indices=[11]),
    decode_head=dict(
        type='CLIPHeadSeg',
        img_size=512,
        in_channels=512,
        channels=512,
        num_classes=20,
        num_layers=3,
        num_heads=8,
        use_stages=1,
        embed_dims=512,
        seen_idx=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19
        ],
        all_idx=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19
        ],
        use_proj=False,
        loss_decode=dict(
            type='SegLossPlus',
            num_classes=20,
            dec_layers=3,
            mask_weight=100.0,
            dice_weight=1.0,
            loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(426, 426)),
    pretrained_text='/data/ljh/pretrained/ViT-B-16.pt',
    base_class=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    ],
    novel_class=[],
    both_class=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    ],
    ft_backbone=False,
    exclude_key='prompt',
    load_text_embedding=None)
dataset_type = 'ZeroPascalVOCDataset20'
data_root = '/data/ljh/PASCAL_VOC_2012/VOCdevkit/VOC2012'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, min_size=512),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='ZeroPascalVOCDataset20',
        data_root='/data/ljh/PASCAL_VOC_2012/VOCdevkit/VOC2012',
        img_dir='JPEGImages',
        ann_dir='SegmentationClassAug',
        split='list/train_aug.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='ZeroPascalVOCDataset20',
        data_root='/data/ljh/PASCAL_VOC_2012/VOCdevkit/VOC2012',
        img_dir='JPEGImages',
        ann_dir='SegmentationClassAug',
        split='list/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True, min_size=512),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='ZeroPascalVOCDataset20',
        data_root='/data/ljh/PASCAL_VOC_2012/VOCdevkit/VOC2012',
        img_dir='JPEGImages',
        ann_dir='SegmentationClassAug',
        split='list/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True, min_size=512),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters = True
optimizer = dict(
    type='AdamW',
    lr=2e-05,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=10.0),
            text_encoder=dict(lr_mult=0.0),
            norm=dict(decay_mult=0.0),
            ln=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=20001)
evaluation = dict(interval=1000, metric='mIoU', save_best='U_mIoU')
base_class = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
]
novel_class = []
both_class = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
]
num_classes = 20
pretrained = '/data/ljh/pretrained/ViT-B-16.pt'
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-06,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06)
work_dir = '/home/ljh/SemanticSegmentation/_LIKE_ZegCLIP/output/voc12/cpt_debug24'
gpu_ids = range(0, 1)
