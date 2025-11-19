CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 \
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
tools/test.py configs/adatad/videomae/e2e_thumos_videomae_s_768x1_160_LongLoRA_distillation_auto.py
