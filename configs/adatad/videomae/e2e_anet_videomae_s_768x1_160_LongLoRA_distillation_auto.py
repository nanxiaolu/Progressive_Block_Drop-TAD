_base_ = [
    "../../_base_/datasets/activitynet-1.3/e2e_resize_768_1x224x224.py",  # dataset config
    "../../_base_/models/actionformer_student.py",  # model config
    "../../_base_/models/actionformer_teacher.py",
]

resize_length = 192
scale_factor = 4
chunk_num = resize_length * scale_factor // 16  # 768/16=48 chunks, since videomae takes 16 frames as input
dataset = dict(
    train=dict(
        resize_length=resize_length,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize", scale_factor=scale_factor),  # load 192x4=768 frames
            dict(type="mmaction.DecordDecode"),
            # dict(type="mmaction.RawFrameDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
        sample_ratio=1.0,
    ),
    val=dict(
        resize_length=resize_length,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize", scale_factor=scale_factor),  # load 192x4=768 frames
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        resize_length=resize_length,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize", scale_factor=scale_factor),  # load 192x4=768 frames
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
    inference_train=dict(
        resize_length=resize_length,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize", scale_factor=scale_factor),  # load 192x4=768 frames
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

cal_loss = dict(
    cal_loss_mid=False, # 1
    cal_loss_out=True, # 2
    cal_loss_mid_rel=False, # 3
    cal_loss_out_rel=False, # 4
    cal_cls_loss=True, # 5 
    cal_reg_loss=True, # 6
    use_dkd_loss=False, # kl or dkd
)

teacher_model = dict(
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerTeacher",
            img_size=224,
            patch_size=16,
            embed_dims=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            drop_path_rate=0.1,
            norm_cfg=dict(type="LN", eps=1e-6),
            return_feat_map=True,
            with_cp=False,  # enable activation checkpointing
            drop_idx=[],
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                dict(type="Interpolate", keys=["feats"], size=resize_length),
            ],
            norm_eval=False,  # also update the norm layers
            freeze_backbone=False,  # unfreeze the backbone
            ckpt_path="./exps/anet/adatad/e2e_actionformer_videomae_s_192x4_160_longlora/0.001_5e-05/gpu2_id0/checkpoint/merge.pth", # you need to change it to your own path
        ),
    ),
    projection=dict(
        in_channels=384,
        out_channels=256,
        attn_cfg=dict(n_mha_win_size=-1),
        use_abs_pe=True,
        max_seq_len=resize_length,
    ),
    neck=dict(in_channels=256, out_channels=256),
    rpn_head=dict(
        in_channels=256,
        feat_channels=256,
        num_classes=1,
        label_smoothing=0.1,
        loss_weight=2.0,
        loss_normalizer=200,
    ),
    cal_cls_loss=cal_loss['cal_cls_loss'], # 5
    cal_reg_loss=cal_loss["cal_reg_loss"], # 6
)

model = dict(
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerLongLoRAStudent",
            img_size=224,
            patch_size=16,
            embed_dims=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            drop_path_rate=0.1,
            norm_cfg=dict(type="LN", eps=1e-6),
            return_feat_map=True,
            with_cp=True,  # enable activation checkpointing
            lora_rank=384 // 4,
            drop_idx=[],
            cal_loss_mid=cal_loss["cal_loss_mid"], # 1
            cal_loss_out=cal_loss["cal_loss_out"], # 2
            cal_loss_mid_rel=cal_loss["cal_loss_mid_rel"], # 3
            cal_loss_out_rel=cal_loss["cal_loss_out_rel"], # 4
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                dict(type="Interpolate", keys=["feats"], size=resize_length),
            ],
            norm_eval=False,  # also update the norm layers
            freeze_backbone=False,  # unfreeze the backbone
            ckpt_path="./exps/anet/adatad/e2e_actionformer_videomae_s_192x4_160_longlora/0.001_5e-05/gpu2_id0/checkpoint/merge.pth", # you need to change it to your own path
            ),
    ),
    projection=dict(
        in_channels=384,
        out_channels=256,
        attn_cfg=dict(n_mha_win_size=-1),
        use_abs_pe=True,
        max_seq_len=resize_length,
    ),
    neck=dict(in_channels=256, out_channels=256),
    rpn_head=dict(
        in_channels=256,
        feat_channels=256,
        num_classes=1,
        label_smoothing=0.1,
        loss_weight=2.0,
        loss_normalizer=200,
    ),
    soft_loss=dict(
        cls_loss=("dkd" if cal_loss['use_dkd_loss'] else "kl"),
        reg_loss=dict(type="DIOULoss"),
    ),
    temperature=3.5,
)

# anet batch_size = 4 * gpu_num
solver = dict(
    train=dict(batch_size=16, num_workers=8),
    val=dict(batch_size=16, num_workers=8),
    test=dict(batch_size=16, num_workers=8),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
    weight=dict(
        loss_mid=0., # 1
        loss_out=1.0, # 2
        loss_mid_rel=0., # 3
        loss_out_rel=0., # 4
        loss_cls_kd=1.0, # 5
        loss_reg_kd=1.0, # 6
        cls_loss=1., # 7
        reg_loss=1., # 8
    ),
    dynamic_teacher=True,
    drop_type="train_mAP", # "train_mAP", "train_loss", "block_mse"
)

# anet 4 gpu 1e-4
custom_lr=1e-4
optimizer = dict(
    type="AdamW",
    lr=1e-3,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="lora", lr=custom_lr, weight_decay=0.05), dict(name="norm", lr=custom_lr, weight_decay=0.05), dict(name="patch_embed", lr=custom_lr, weight_decay=0.05)],
        exclude=["backbone"],
    ),
)

sub_scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=10)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=15)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,
        max_seg_num=100,
        min_score=0.001,
        multiclass=False,
        voting_thresh=0.9,  #  set 0 to disable
    ),
    external_cls=dict(
        type="CUHKANETClassifier",
        path="data/activitynet-1.3/classifiers/cuhk_val_simp_7.json",
        topk=2,
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=99999,
    checkpoint_interval=-1,
    val_loss_interval=-1,

    sub_val_first_start_epoch=4,
    sub_val_first_eval_interval=1,
    sub_val_second_start_epoch=4,
    sub_val_second_eval_interval=1,
    sub_end_epoch=8,

    full_val_first_start_epoch=2,
    full_val_first_eval_interval=3,
    full_val_second_start_epoch=8,
    full_val_second_eval_interval=1,
    full_end_epoch=15,
)

work_dir = f"exps/anet/adatad/longlora_768x1_160_distillation_auto"
