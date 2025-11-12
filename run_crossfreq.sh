#!/usr/bin/env bash
set -e

# === Edit these paths before running ===
DATASET_NAME="imagenet_subset"          # or "cifar10"
DATA_ROOT="/home/UNT/ak2102/DeepLearning_project/vit_vs_resnet/data/imagenet_subset"      # e.g., /home/you/data/imagenet_subset
OUT_DIR="checkpoints_crossfreq"
PROJECT="crossfreq-vit"

# Example hyperparams for ViT-B/16 fine-tuning on ImageNet-mini
BATCH_SIZE=128
EPOCHS=30
LR=5e-4
WD=0.05
MIXUP=0.0
# AUG="--augment"       # remove to disable augmentations

# CrossFreq settings
FUSION_AT=6
LF_TOKENS=4
LF_CUTOFF=0.15
HF_BINS=16

# Uncomment to enable ArcFace (usually off for standard classification)
ARCFLAGS="--use_arcface --arc_s 64.0 --arc_m 0.5"

python run_experiment_crossfreq_vit.py \
  --dataset_name "${DATASET_NAME}" \
  --data_root "${DATA_ROOT}" \
  --train_frac 1.0 \
  --batch_size ${BATCH_SIZE} \
  --num_workers 8 \
  ${AUG} \
  --fusion_at ${FUSION_AT} \
  --lf_tokens ${LF_TOKENS} \
  --lf_cutoff ${LF_CUTOFF} \
  --hf_bins ${HF_BINS} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --weight_decay ${WD} \
  --mixup_alpha ${MIXUP} \
  --project "${PROJECT}" \
  --out_dir "${OUT_DIR}" \
  ${ARCFLAGS}
