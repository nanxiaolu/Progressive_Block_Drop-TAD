_base_ = ["e2e_thumos_videomae_s_768x1_160_LongLoRA_distillation_auto.py"]

teacher_model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=1024,
            depth=24,
            num_heads=16,
            lora_rank=1024 // 4,
        ),
        custom=dict(
            ckpt_path="/data-store/chenxiaoyong/lora_opentad/exps/thumos/adatad/e2e_actionformer_videomae_l_768x1_160_Longlora/5e-05/gpu2_id0/checkpoint/merge.pth", # you need to change it to your own path
        ),
    ),
    projection=dict(in_channels=1024),
)

model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=1024,
            depth=24,
            num_heads=16,
            lora_rank=1024 // 4,
        ),
        custom=dict(
            ckpt_path="/data-store/chenxiaoyong/lora_opentad/exps/thumos/adatad/e2e_actionformer_videomae_l_768x1_160_Longlora/5e-05/gpu2_id0/checkpoint/merge.pth", # you need to change it to your own path
        ),
    ),
    projection=dict(in_channels=1024),
)

sub_scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=80)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=80)
workflow = dict(
    sub_val_first_start_epoch=10,
    sub_val_first_eval_interval=5,
    sub_val_second_start_epoch=40,
    sub_val_second_eval_interval=1,
    sub_end_epoch=60,

    full_val_first_start_epoch=10,
    full_val_first_eval_interval=5,
    full_val_second_start_epoch=40,
    full_val_second_eval_interval=1,
    full_end_epoch=60,
)

custom_lr = 1e-4
optimizer = dict(backbone=dict(custom=[dict(name="lora", lr=custom_lr, weight_decay=0.05), dict(name="norm", lr=custom_lr, weight_decay=0.05), dict(name="patch_embed", lr=custom_lr, weight_decay=0.05)]))

work_dir = f"exps/thumos/adatad/e2e_actionformer_videomae_l_768x1_160_longlora_dist_auto"
