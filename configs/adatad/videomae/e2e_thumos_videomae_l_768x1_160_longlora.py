_base_ = ["e2e_thumos_videomae_s_768x1_160_LongLoRA.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=1024,
            depth=24,
            num_heads=16,
            lora_rank=1024 // 4,
        ),
        custom=dict(pretrain="/opentad-pretrained/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth"), # you need to change it to your own path
    ),
    projection=dict(in_channels=1024),
)

custom_lr = 5e-5
optimizer = dict(backbone=dict(custom=[dict(name="lora", lr=custom_lr, weight_decay=0.05), dict(name="norm", lr=custom_lr, weight_decay=0.05), dict(name="patch_embed", lr=custom_lr, weight_decay=0.05)]))

work_dir = f"exps/thumos/adatad/e2e_actionformer_videomae_l_768x1_160_Longlora"