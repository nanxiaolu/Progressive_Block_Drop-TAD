# Temporal Action Detection Model Compression by Progressive Block Drop
## Introduction
This repository contains the official code for the paper “Temporal Action Detection Model Compression by Progressive Block Drop,” which has been accepted by CVPR 2025. The paper can be found at [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Temporal_Action_Detection_Model_Compression_by_Progressive_Block_Drop_CVPR_2025_paper.html) and [arXiv](https://arxiv.org/abs/2503.16916), and the video is available on [YouTube](https://www.youtube.com/watch?v=QhSE8sLffOg).  

The paper proposes **a task- and model-agnostic model compression method**, and validates its effectiveness on the temporal action detection task.

The code is adapted from the work [OpenTAD](https://github.com/sming256/OpenTAD), with data preparation, environment setup, code structure, and running commands largely unchanged. This repository supports the Thumos dataset and (optional) the ActivityNet dataset.

![Model Framework](docs/figures/paper_cvpr2025_block_drop.png)

## 🛠️ Installation

Please prepare the environment, data, and models as described in [install.md](docs/en/install.md).

## Commands
Training command (default THUMOS dataset):
```
bash tools/train.sh
```
Testing command (default THUMOS dataset):
```
bash tools/test.sh
```
LoRA merge:
```
python tools/model_converters/merge_model.py # Modify input parameters inside the file as indicated by comments
```
## Key Code Files:
```
tools/train_distillation_auto.py # Main training script, where eval_drop_one_block decides which block to drop

opentad/models/detectors/actionformer_student.py # Outer layer of the student model for distillation

opentad/models/detectors/actionformer_teacher.py # Outer layer of the teacher model for distillation

opentad/models/backbones/vit_longlora_student.py # Inner layer of the student model for distillation

opentad/models/backbones/vit_teacher.py # Inner layer of the teacher model for distillation
```
## (Optional) Deeper Model Compression
videomae-small has 12 layers, and videomae-large has 24 layers.  
Compression is more effective on deeper, larger models. To try compression on videomae-large, follow the instructions in [train.sh](tools/train.sh):
1. Download the pretrained videomae-large weights [Download Link](https://github.com/sming256/OpenTAD/tree/main/configs/adatad#prepare-the-pretrained-videomae-checkpoints)
2. Fine-tune a full uncompressed baseline on the Thumos dataset using [config file](configs/adatad/videomae/e2e_thumos_videomae_l_768x1_160_longlora.py), modifying one place as indicated (# you need to change it to your own path)
3. Perform compression using [config file](configs/adatad/videomae/e2e_thumos_videomae_l_768x1_160_LongLoRA_distillation_auto.py), modifying two places as indicated (# you need to change it to your own path)

## (Optional) Other Datasets
This repository also provides configuration files for the ActivityNet dataset. To try compression on videomae-small, follow these steps:
1. Download the pretrained videomae-small weights [Download Link](https://github.com/sming256/OpenTAD/tree/main/configs/adatad#prepare-the-pretrained-videomae-checkpoints)
2. Fine-tune a full uncompressed baseline on the ActivityNet dataset using [config file](configs/adatad/videomae/e2e_anet_videomae_s_768x1_160_LongLoRA.py), modifying one place as indicated (# you need to change it to your own path). We also provide our implementation [here](https://pan.baidu.com/s/1Z2fkI_24vX6hchsJlQhZnA?pwd=tpbd).

3. Perform compression using [config file](configs/adatad/videomae/e2e_anet_videomae_s_768x1_160_LongLoRA_distillation_auto.py), modifying two places as indicated (# you need to change it to your own path). We also provide our implementation [here](https://pan.baidu.com/s/1xFP1lL5-AxwiwYPGzE34ng?pwd=tpbd).



## Contact
If you have any questions regarding the code or paper, please contact us via the email on the paper homepage.