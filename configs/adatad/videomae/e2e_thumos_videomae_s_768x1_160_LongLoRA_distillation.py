_base_ = [
    "../../_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py",  # dataset config
    "../../_base_/models/actionformer_student.py",  # model config
    "../../_base_/models/actionformer_teacher.py",
]

window_size = 768
scale_factor = 1
chunk_num = window_size * scale_factor // 16  # 768/16=48 chunks, since videomae takes 16 frames as input
dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
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
    ),
    val=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
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
    cal_loss_out=False, # 2
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
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
            norm_eval=False,  # also update the norm layers
            freeze_backbone=False,  # unfreeze the backbone
            # ckpt_path="/data-store/chenxiaoyong/lora_opentad/exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_Longlora/merge_model/gpu2_id0/checkpoint/best.pth",
            ckpt_path="/data-store/chenxiaoyong/lora_opentad/exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_dist/0_0_0_0_1_1/1_0 1_0 1_0 1_0 1_0 1_0 1_0 1_0/drop_4_0.0002/gpu2_id0/checkpoint/merge.pth",
        ),
    ),
    projection=dict(
        in_channels=384,
        max_seq_len=window_size,
        attn_cfg=dict(n_mha_win_size=-1),
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
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
            norm_eval=False,  # also update the norm layers
            freeze_backbone=False,  # unfreeze the backbone
            # ckpt_path="/data-store/chenxiaoyong/lora_opentad/exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_Longlora/merge_model/gpu2_id0/checkpoint/best.pth",
            ckpt_path="/data-store/chenxiaoyong/lora_opentad/exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_dist/0_0_0_0_1_1/1_0 1_0 1_0 1_0 1_0 1_0 1_0 1_0/drop_4_0.0002/gpu2_id0/checkpoint/merge.pth",
        ),
    ),
    projection=dict(
        in_channels=384,
        max_seq_len=window_size,
        attn_cfg=dict(n_mha_win_size=-1),
    ),
    soft_loss=dict(
        cls_loss=("dkd" if cal_loss['use_dkd_loss'] else "kl"),
        reg_loss=dict(type="DIOULoss"),
    ),
)

solver = dict(
    train=dict(batch_size=2, num_workers=2, prefetch_factor=2),
    val=dict(batch_size=2, num_workers=2, prefetch_factor=2),
    test=dict(batch_size=2, num_workers=2, prefetch_factor=2),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
    weight=dict(
        loss_mid=1.0, # 1
        loss_out=1.0, # 2
        loss_mid_rel=1., # 3
        loss_out_rel=1., # 4
        loss_cls_kd=1.0, # 5
        loss_reg_kd=1.0, # 6
        cls_loss=1., # 7
        reg_loss=1., # 8
    )
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="lora", lr=2e-4, weight_decay=0.05), dict(name="norm", lr=5e-5, weight_decay=0.05), dict(name="patch_embed", lr=2e-4, weight_decay=0.05)],
        exclude=["backbone"],
    ),
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=60)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,
        max_seg_num=2000,
        multiclass=True,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=99999,
    checkpoint_interval=-1,
    val_loss_interval=-1,
    val_first_start_epoch=10,
    val_first_eval_interval=5,
    val_second_start_epoch=40,
    val_second_eval_interval=1,
    end_epoch=60,
)

# work_dir = (f"exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_dist/"
#             f"{'_'.join(str(int(i)) for _, i in cal_loss.items() if type(i) == bool)}/"
#             f"{' '.join(str(i).replace('.', '_') for _, i in solver['weight'].items())}/" + 
#             f"drop_{'_'.join(str(i) for i in model['backbone']['backbone']['drop_idx'])}_{optimizer['backbone']['custom'][0]['lr']}_step_teacher")
# work_dir = f"exps/thumos/adatad/debug"
work_dir = "/mnt/data/chenxiaoyong/tad_accelerate/exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_Longlora/merge_model"
# print(work_dir)
