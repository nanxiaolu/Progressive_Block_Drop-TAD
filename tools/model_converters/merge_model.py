import torch

def MergeLoRA(ckpt_path, update_model_path, depth):
    pretrained_state_dict = torch.load(ckpt_path)

    for i in range(depth):
        try:
            pretrained_state_dict['state_dict'][f'module.backbone.model.backbone.blocks.{i}.attn.qkv.weight'] = (
                pretrained_state_dict['state_dict'][f'module.backbone.model.backbone.blocks.{i}.attn.qkv.weight'] + 
                (pretrained_state_dict['state_dict'][f'module.backbone.model.backbone.blocks.{i}.attn.lora_qkv.A'] @ pretrained_state_dict['state_dict'][f'module.backbone.model.backbone.blocks.{i}.attn.lora_qkv.B']).T
            )
            del pretrained_state_dict['state_dict'][f'module.backbone.model.backbone.blocks.{i}.attn.lora_qkv.A'], pretrained_state_dict['state_dict'][f'module.backbone.model.backbone.blocks.{i}.attn.lora_qkv.B']

            pretrained_state_dict['state_dict'][f'module.backbone.model.backbone.blocks.{i}.attn.proj.weight'] = (
                pretrained_state_dict['state_dict'][f'module.backbone.model.backbone.blocks.{i}.attn.proj.weight'] + 
                (pretrained_state_dict['state_dict'][f'module.backbone.model.backbone.blocks.{i}.attn.lora_proj.A'] @ pretrained_state_dict['state_dict'][f'module.backbone.model.backbone.blocks.{i}.attn.lora_proj.B']).T
            )
            del pretrained_state_dict['state_dict'][f'module.backbone.model.backbone.blocks.{i}.attn.lora_proj.A'], pretrained_state_dict['state_dict'][f'module.backbone.model.backbone.blocks.{i}.attn.lora_proj.B']

            pretrained_state_dict['state_dict_ema'][f'backbone.model.backbone.blocks.{i}.attn.qkv.weight'] = (
                pretrained_state_dict['state_dict_ema'][f'backbone.model.backbone.blocks.{i}.attn.qkv.weight'] + 
                (pretrained_state_dict['state_dict_ema'][f'backbone.model.backbone.blocks.{i}.attn.lora_qkv.A'] @ pretrained_state_dict['state_dict_ema'][f'backbone.model.backbone.blocks.{i}.attn.lora_qkv.B']).T
            )        
            del pretrained_state_dict['state_dict_ema'][f'backbone.model.backbone.blocks.{i}.attn.lora_qkv.A'], pretrained_state_dict['state_dict_ema'][f'backbone.model.backbone.blocks.{i}.attn.lora_qkv.B']

            pretrained_state_dict['state_dict_ema'][f'backbone.model.backbone.blocks.{i}.attn.proj.weight'] = (
                pretrained_state_dict['state_dict_ema'][f'backbone.model.backbone.blocks.{i}.attn.proj.weight'] + 
                (pretrained_state_dict['state_dict_ema'][f'backbone.model.backbone.blocks.{i}.attn.lora_proj.A'] @ pretrained_state_dict['state_dict_ema'][f'backbone.model.backbone.blocks.{i}.attn.lora_proj.B']).T
            )
            del pretrained_state_dict['state_dict_ema'][f'backbone.model.backbone.blocks.{i}.attn.lora_proj.A'], pretrained_state_dict['state_dict_ema'][f'backbone.model.backbone.blocks.{i}.attn.lora_proj.B']
        except:
            pass

    torch.save(pretrained_state_dict, update_model_path)
    print("merge done!")

if __name__ == '__main__':
    # the first arg is the lora checkpoint path
    # the second arg is the merged checkpoint path
    # the third arg is the depth of the backbone, for videomae_s, depth=12, for videomae_l, depth=24
    MergeLoRA("./exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_Longlora/merge_model/gpu2_id0/checkpoint/best.pth",
            "./exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_Longlora/merge_model/gpu2_id0/checkpoint/merge.pth",
            12)
    
