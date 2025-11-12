#!/bin/bash

# Where your data is stored.
# For CIFAR10, torchvision will download into this path automatically.
# For imagenet_subset, you should have:
#   data/imagenet_subset/train/<class>/*.jpg
#   data/imagenet_subset/val/<class>/*.jpg
#   data/imagenet_subset/test/<class>/*.jpg
DATA_ROOT="./data"

# Common training settings
EPOCHS=50
BATCH_SIZE=128
MIXUP_ALPHA=0.0        # set 0.0 if you want to disable mixup
AUGMENT=False            # set False if you want no data augmentation
SEED=42
LR=5e-5
WEIGHT_DECAY=0.05
PATIENCE=5              # early stopping patience (in epochs)
WARMUP_STEPS=500        # warmup steps for cosine schedule

# Fractions of training data to try
FRACTIONS=(0.01 0.1 1.0)

# Datasets to try
# DATASETS=("cifar10" "imagenet_subset")
DATASETS=("imagenet_subset")

# Models to try
# MODELS=("resnet152" "vit_b16")
MODELS=("vit_b16")

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for frac in "${FRACTIONS[@]}"; do

      echo "============================================================"
      echo "Running model=$model dataset=$dataset train_frac=$frac"
      echo "============================================================"

      # choose correct root path for this dataset
      if [ "$dataset" = "cifar10" ]; then
        ROOT_PATH="$DATA_ROOT/cifar10"
      else
        ROOT_PATH="$DATA_ROOT/imagenet_subset/"
      fi

      python run_experiment.py \
        --dataset $dataset \
        --data_root $ROOT_PATH \
        --model $model \
        --train_frac $frac \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --mixup_alpha $MIXUP_ALPHA \
        --augment $AUGMENT \
        --seed $SEED \
        --lr $LR \
        --weight_decay $WEIGHT_DECAY \
        --patience $PATIENCE \
        --warmup_steps $WARMUP_STEPS \
        --wandb_project vit_vs_resnetv2

    done
  done
done

