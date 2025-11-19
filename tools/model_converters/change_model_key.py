import torch

def changeModelKey(ckpt_path, update_model_path):
    pretrained_state_dict = torch.load(ckpt_path)

    updated_state_dict = {}

    for key, value in pretrained_state_dict.items():
        if "mlp.layers.0.0.weight" in key:
            new_key = key.replace("mlp.layers.0.0.weight", "mlp.layers.0.0.linear.weight")
        elif "mlp.layers.0.0.bias" in key:
            new_key = key.replace("mlp.layers.0.0.bias", "mlp.layers.0.0.linear.bias")
        elif "backbone.blocks" in key and "mlp.layers.1.weight" in key:
            new_key = key.replace("mlp.layers.1.weight", "mlp.layers.1.linear.weight")
        elif "backbone.blocks" in key and "mlp.layers.1.bias" in key:
            new_key = key.replace("mlp.layers.1.bias", "mlp.layers.1.linear.bias")
        else:
            new_key = key
        
        updated_state_dict[new_key] = value
    torch.save(updated_state_dict, update_model_path)

changeModelKey('opentad-pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth', 'pretrain_model/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_lora.pth')