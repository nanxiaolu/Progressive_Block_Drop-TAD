# for videomae-small, single GPU
CUDA_VISIBLE_DEVICES=0 \
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
tools/train_distillation_auto.py \
configs/adatad/videomae/e2e_thumos_videomae_s_768x1_160_LongLoRA_distillation_auto.py

# for videomae-large
# download pretrain model on OpenTAD repository first (https://github.com/sming256/OpenTAD/tree/main/configs/adatad#prepare-the-pretrained-videomae-checkpoints), and you should finetuning a baseline model:
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
# tools/train.py \
# configs/adatad/videomae/e2e_thumos_videomae_l_768x1_160_longlora.py

# then use the obtained model as the teacher to distill the LongLoRA model:
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
# tools/train_distillation_auto.py \
# configs/adatad/videomae/e2e_thumos_videomae_l_768x1_160_LongLoRA_distillation_auto.py