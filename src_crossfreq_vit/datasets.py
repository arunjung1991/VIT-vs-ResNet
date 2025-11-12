# src_crossfreq_vit/datasets.py
# Dataloaders for CIFAR-10 and ImageNet(-mini) with consistent 224×224 eval sizing.

import os
import math
from typing import Dict, Tuple, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

# ImageNet normalization used by torchvision ViT/ResNet weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# -------------------------
# Transforms
# -------------------------

def build_train_transforms_cifar10() -> transforms.Compose:
    """
    CIFAR-10 is 32×32. We apply standard augments at 32×32 then upsample
    to 224×224 to match ImageNet-pretrained backbones.
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_train_transforms_imagenet() -> transforms.Compose:
    """
    Typical ImageNet training augmentation to 224×224.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transforms() -> transforms.Compose:
    """
    Validation / test transforms (deterministic).
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# -------------------------
# Helpers
# -------------------------

def _subset_by_fraction(dataset, fraction: float, *, shuffle: bool = True, seed: int = 42) -> Subset:
    """
    Keep only a fraction (e.g. 0.1 = 10%) of a dataset.
    """
    assert 0 < fraction <= 1.0, "train_frac must be in (0, 1]."

    n_total = len(dataset)
    n_keep = max(1, int(math.floor(n_total * fraction)))

    indices = np.arange(n_total)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    chosen_idx = indices[:n_keep]
    return Subset(dataset, chosen_idx.tolist())


def _split_subset(dataset, fractions, seed: int = 1234):
    """
    Deterministically split a dataset (e.g., ImageFolder) into len(fractions) Subsets.
    fractions must sum to 1.0. Example: [0.5, 0.5].
    """
    assert abs(sum(fractions) - 1.0) < 1e-6, "fractions must sum to 1.0"
    n = len(dataset)
    idxs = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idxs)

    splits = []
    start = 0
    for frac in fractions:
        count = int(round(frac * n))
        end = start + count
        part = idxs[start:end]
        splits.append(Subset(dataset, part.tolist()))
        start = end
    return splits


# -------------------------
# CIFAR-10 splits
# -------------------------

def _get_cifar10_splits(
    data_root: str,
    train_frac: float,
    batch_size: int,
    num_workers: int,
    augment: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Build CIFAR-10 train/val/test loaders.

    Strategy:
      • Use 45k of the original train as train, 5k as val (deterministic).
      • Optionally apply strong aug to train.
      • Always use eval transforms for val/test.
      • Subsample train by train_frac.
    """
    train_tf = build_train_transforms_cifar10() if augment else build_eval_transforms()
    eval_tf  = build_eval_transforms()

    # Two views so val is clean (no train-time augs)
    full_train_aug = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    full_train_eval = datasets.CIFAR10(root=data_root, train=True, download=False, transform=eval_tf)

    # 50k total -> 45k/5k split (random_split shuffles deterministically with generator)
    train_len, val_len = 45_000, 5_000
    generator = torch.Generator().manual_seed(1234)
    train_subset_aug, val_subset_tmp = random_split(full_train_aug, [train_len, val_len], generator=generator)

    # Rebuild val subset with eval transforms using the same indices
    val_indices = val_subset_tmp.indices
    val_set = Subset(full_train_eval, val_indices)

    # Subsample the train split by fraction
    train_set = _subset_by_fraction(train_subset_aug, fraction=train_frac, shuffle=True, seed=42)

    # CIFAR10 test (always eval transforms)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=eval_tf)

    num_classes = 10

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes


# -------------------------
# ImageNet(-mini) splits
# -------------------------

def _get_imagenet_subset_splits(
    data_root: str,
    train_frac: float,
    batch_size: int,
    num_workers: int,
    augment: bool,
):
    """
    Expects ImageNet(-mini) layout:
        data_root/train/<class>/*.jpg
        data_root/val/<class>/*.jpg

    There is NO explicit 'test/' directory. We:
      • Load train/ with train transforms (optionally augmented).
      • Load val/ with eval transforms.
      • Split the original val/ 50/50 into our val and test (deterministic).
      • Subsample train/ by train_frac.
    """
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")

    train_tf = build_train_transforms_imagenet() if augment else build_eval_transforms()
    eval_tf  = build_eval_transforms()

    full_train = datasets.ImageFolder(train_dir, transform=train_tf)
    full_val_eval = datasets.ImageFolder(val_dir, transform=eval_tf)

    num_classes = len(full_train.classes)

    # Split original 'val' into val/test halves deterministically
    val_subset, test_subset = _split_subset(full_val_eval, fractions=[0.5, 0.5], seed=1234)

    # Subsample train by fraction
    train_sub = _subset_by_fraction(full_train, fraction=train_frac, shuffle=True, seed=42)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes


# -------------------------
# Public entry
# -------------------------

def get_dataloaders(
    dataset_name: Literal["cifar10", "imagenet_subset"],
    data_root: str,
    train_frac: float,
    batch_size: int,
    num_workers: int = 4,
    augment: bool = False,
) -> Dict:
    """
    Return a dict with 'train', 'val', 'test', 'num_classes' for the chosen dataset.
    """
    name = dataset_name.lower()

    if name == "cifar10":
        train_loader, val_loader, test_loader, num_classes = _get_cifar10_splits(
            data_root=data_root,
            train_frac=train_frac,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
        )
    elif name in ("imagenet_subset", "imagenet-subset", "imagenet"):
        train_loader, val_loader, test_loader, num_classes = _get_imagenet_subset_splits(
            data_root=data_root,
            train_frac=train_frac,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
        )
    else:
        raise ValueError("dataset_name must be 'cifar10' or 'imagenet_subset'.")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "num_classes": num_classes,
    }
