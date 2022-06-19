# 导入mmcls模块
custom_imports=dict(imports='mmcls.models', allow_failed_imports=False)

# model settings
model = dict(
    type='NanoDet',
    backbone=dict(
        type='mmcls.ShuffleNetV2',
        out_indices=(0, 1, 2),
        widen_factor=1.0,
        act_cfg=dict(type='LeakyReLU')),
    neck=dict(
        type='NANOPAN',
        in_channels=[116, 232, 464],
        out_channels=96,
        start_level=0,
        num_outs=3),
    bbox_head=dict(
        type='NanoDetHead',
        share_cls_reg=True,
        num_classes=80,
        in_channels=96,
        stacked_convs=2,
        feat_channels=96,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=5,
            scales_per_octave=1,
            strides=[8, 16, 32]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=7,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
