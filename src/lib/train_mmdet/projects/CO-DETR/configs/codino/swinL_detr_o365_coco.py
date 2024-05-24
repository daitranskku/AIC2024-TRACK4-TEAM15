_base_ = ['r50_detr.py']

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'  # noqa

# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    query_head=dict(
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
        transformer=dict(encoder=dict(with_cp=6))))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1280), (512, 1280), (544, 1280), (576, 1280),
                            (608, 1280), (640, 1280), (672, 1280), (704, 1280),
                            (736, 1280), (768, 1280), (800, 1280), (832, 1280),
                            (864, 1280), (896, 1280), (928, 1280), (960, 1280),
                            (992, 1280), (1024, 1280), (1056, 1280),
                            (1088, 1280), (1120, 1280), (1152, 1280),
                            (1184, 1280), (1216, 1280), (1248, 1280),
                            (1280, 1280), (1312, 1280), (1344, 1280),
                            (1376, 1280), (1408, 1280), (1440, 1280),
                            (1472, 1280), (1504, 1280), (1536, 1280)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1280), (512, 1280), (544, 1280), (576, 1280),
                            (608, 1280), (640, 1280), (672, 1280), (704, 1280),
                            (736, 1280), (768, 1280), (800, 1280), (832, 1280),
                            (864, 1280), (896, 1280), (928, 1280), (960, 1280),
                            (992, 1280), (1024, 1280), (1056, 1280),
                            (1088, 1280), (1120, 1280), (1152, 1280),
                            (1184, 1280), (1216, 1280), (1248, 1280),
                            (1280, 1280), (1312, 1280), (1344, 1280),
                            (1376, 1280), (1408, 1280), (1440, 1280),
                            (1472, 1280), (1504, 1280), (1536, 1280)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=1, num_workers=1, dataset=dict(pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1280, 1280), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

optim_wrapper = dict(optimizer=dict(lr=1e-4))

max_epochs = 32
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8],
        gamma=0.1)
]
work_dir = './work_dirs/final_detr_o365_1280'