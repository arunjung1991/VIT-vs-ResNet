# 🧠 ViT vs ResNet — Baselines and CrossFreq-ViT (Frequency-Aware Vision Transformer)

This project compares **ResNet-152** and **ViT-B/16** for image classification and introduces **CrossFreq-ViT**, a frequency-aware transformer that injects low- and high-frequency cues into intermediate transformer layers to enhance feature representation and generalization.

Supported datasets: **CIFAR-10** and **ImageNet Subset (ImageNet-mini)**

---

## ⚙️ Environment Setup

```bash
# Create environment
conda create -n vit-vs-resnet python=3.10 -y
conda activate vit-vs-resnet

# Install PyTorch (CUDA 12.x example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

---

## 📂 Dataset Setup

### CIFAR-10
Automatically downloaded by `torchvision` into `data/cifar10`.

### ImageNet Subset (ImageNet-mini)
Expected directory structure:
```
data/
└─ imagenet_subset/
   ├─ train/
   │   ├─ classA/*.jpg
   │   └─ classB/*.jpg
   ├─ val/
   │   ├─ classA/*.jpg
   └─ test/
       ├─ classA/*.jpg
```
If `test/` is not available, duplicate `val/` as a placeholder.

---

## 🚀 Baseline Training (ResNet-152 / ViT-B/16)

**Script:** `run_all.sh`

```bash
#!/bin/bash
DATA_ROOT="./data"
EPOCHS=50
BATCH_SIZE=128
MIXUP_ALPHA=0.0
AUGMENT=False
SEED=42
LR=5e-5
WEIGHT_DECAY=0.05
PATIENCE=5
WARMUP_STEPS=500

FRACTIONS=(0.01 0.1 1.0)
DATASETS=("imagenet_subset")
MODELS=("vit_b16")

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for frac in "${FRACTIONS[@]}"; do
      if [ "$dataset" = "cifar10" ]; then
        ROOT_PATH="$DATA_ROOT/cifar10"
      else
        ROOT_PATH="$DATA_ROOT/imagenet_subset/"
      fi

      python run_experiment.py         --dataset $dataset         --data_root $ROOT_PATH         --model $model         --train_frac $frac         --batch_size $BATCH_SIZE         --epochs $EPOCHS         --mixup_alpha $MIXUP_ALPHA         --augment $AUGMENT         --seed $SEED         --lr $LR         --weight_decay $WEIGHT_DECAY         --patience $PATIENCE         --warmup_steps $WARMUP_STEPS         --wandb_project vit_vs_resnetv2
    done
  done
done
```

Run:
```bash
bash run_all.sh
```

---

## 🌈 Frequency-Aware ViT (CrossFreq-ViT)

**Script:** `run_crossfreq.sh`

```bash
#!/usr/bin/env bash
set -e

DATASET_NAME="imagenet_subset"
DATA_ROOT="/home/UNT/ak2102/DeepLearning_project/vit_vs_resnet/data/imagenet_subset"
OUT_DIR="checkpoints_crossfreq"
PROJECT="crossfreq-vit"

BATCH_SIZE=128
EPOCHS=30
LR=5e-4
WD=0.05
MIXUP=0.0
# AUG="--augment"  # uncomment to enable augmentations

# Frequency injection parameters
FUSION_AT=6
LF_TOKENS=4
LF_CUTOFF=0.15
HF_BINS=16

# Optional ArcFace
ARCFLAGS="--use_arcface --arc_s 64.0 --arc_m 0.5"

python run_experiment_crossfreq_vit.py   --dataset_name "${DATASET_NAME}"   --data_root "${DATA_ROOT}"   --train_frac 1.0   --batch_size ${BATCH_SIZE}   --num_workers 8   ${AUG}   --fusion_at ${FUSION_AT}   --lf_tokens ${LF_TOKENS}   --lf_cutoff ${LF_CUTOFF}   --hf_bins ${HF_BINS}   --epochs ${EPOCHS}   --lr ${LR}   --weight_decay ${WD}   --mixup_alpha ${MIXUP}   --project "${PROJECT}"   --out_dir "${OUT_DIR}"   ${ARCFLAGS}
```

Run:
```bash
bash run_crossfreq.sh
```

### Key Flags

| Flag | Description |
|------|--------------|
| `--fusion_at` | Transformer layer index where LF/HF tokens are injected |
| `--lf_tokens` | Number of learnable low-frequency tokens |
| `--lf_cutoff` | Radial cutoff for LF/HF separation |
| `--hf_bins` | Number of angular/radial bins for HF features |
| `--use_arcface` | Enables ArcFace classification head |
| `--arc_s`, `--arc_m` | ArcFace scaling and margin parameters |

---

## 🧾 Logging & Reproducibility

### Weights & Biases (wandb)
```bash
pip install wandb
wandb login
```
Set the project name using `--wandb_project` or `--project`.

### Reproducibility
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 🧩 Project Structure

```
.
├─ run_all.sh
├─ run_crossfreq.sh
├─ run_experiment.py
├─ run_experiment_crossfreq_vit.py
├─ src/
│  ├─ datasets.py
│  ├─ models/
│  │  ├─ resnet152.py
│  │  ├─ vit_b16.py
│  │  └─ crossfreq_vit.py
│  ├─ losses/arcface.py
│  ├─ utils/
│  │  ├─ train_utils.py
│  │  └─ schedulers.py
├─ checkpoints/
└─ data/
   ├─ cifar10/
   └─ imagenet_subset/
```

---

## 📈 Outputs

- **Checkpoints** saved in `checkpoints/` and `checkpoints_crossfreq/`
- **Metrics:** Top-1 / Top-5 accuracy, training & validation loss
- **Confusion matrices** and **loss curves**
- **W&B logs** for experiment tracking

---

## 📚 Citations

### APA
> Biewald, L. (2020). *Weights & Biases: Machine learning experiment tracking tool* [Computer software]. Weights & Biases. https://wandb.ai  
> Kaggle. (n.d.). *Kaggle: Your machine learning and data science community* [Website]. https://www.kaggle.com

### IEEE
> L. Biewald, *Weights & Biases*, Weights & Biases Inc., 2020. [Online]. Available: https://wandb.ai  
> Kaggle, *Kaggle: Your Machine Learning and Data Science Community*. [Online]. Available: https://www.kaggle.com

### BibTeX
```bibtex
@misc{wandb,
  author = {Biewald, Lukas},
  title = {Weights and Biases},
  year = {2020},
  howpublished = {\url{https://wandb.ai}}
}

@misc{kaggle,
  author = {{Kaggle}},
  title = {Kaggle: Your Machine Learning and Data Science Community},
  year = {n.d.},
  howpublished = {\url{https://www.kaggle.com}}
}
```

---

## 📜 License

Released under the **MIT License**.  
Free to use for research and educational purposes.

---

## 🙏 Acknowledgements

- **PyTorch** and **TorchVision**  
- **Weights & Biases (wandb.ai)** for experiment tracking  
- **Kaggle** and **ImageNet-mini** for open datasets  
- Open-source contributors for ResNet and ViT architectures

---

## ✅ Quick Commands

| Task | Command |
|------|----------|
| Train Baselines (ResNet / ViT) | `bash run_all.sh` |
| Train Frequency-Aware ViT | `bash run_crossfreq.sh` |

---
