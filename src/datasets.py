# src/datasets.py

import os
import torch
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Literal, Dict
import math
import numpy as np

# Standard ImageNet mean/std used by both ResNet and ViT pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# -------------------------
# Transforms
# -------------------------

def build_train_transforms_cifar10():
    """
    CIFAR is 32x32 originally.
    We'll:
    - pad+random crop (classic 32x32 augmentation),
    - random horizontal flip,
    - then upsample to 224x224 so it matches ImageNet-pretrained backbones,
      and then normalize with ImageNet mean/std.
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_train_transforms_imagenet():
    """
    Typical ImageNet-style augmentation:
    - random resized crop to 224,
    - horizontal flip,
    - normalize.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transforms():
    """
    Validation / test transforms (no randomness).
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

def _subset_by_fraction(dataset, fraction: float, shuffle: bool = True, seed: int = 42):
    """
    Take only a fraction (e.g. 0.1 = 10%) of a dataset.
    """
    assert 0 < fraction <= 1.0, "train_frac must be in (0,1]."

    n_total = len(dataset)
    n_keep = max(1, int(math.floor(n_total * fraction)))

    indices = np.arange(n_total)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    chosen_idx = indices[:n_keep]
    return Subset(dataset, chosen_idx)


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
      - Use first 45k of CIFAR10 train as train, last 5k as val.
      - Apply strong aug to train if augment=True.
      - Always use eval transforms for val/test.
      - Subsample train with train_frac.
    """

    train_transform = build_train_transforms_cifar10() if augment else build_eval_transforms()
    eval_transform  = build_eval_transforms()

    # full CIFAR10 train with either aug or eval-style transform
    full_train_augview = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    # We'll also build another view of CIFAR train with eval transforms
    # so we can carve out a clean val set.
    full_train_evalview = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=eval_transform,
    )

    # 50k total -> 45k train / 5k val split
    train_len = 45000
    val_len = 5000

    # We split indices deterministically
    generator = torch.Generator().manual_seed(1234)
    train_subset_augview, val_subset_evalview = random_split(
        full_train_augview,
        [train_len, val_len],
        generator=generator,
    )

    # random_split shuffles indices internally, so we need to reconstruct
    # val subset using the SAME indices but pulling from the evalview dataset.
    # torch.utils.data.Subset stores .indices
    val_indices = val_subset_evalview.indices
    val_set = Subset(full_train_evalview, val_indices)

    # Now subsample the train set by train_frac
    train_set = _subset_by_fraction(
        train_subset_augview,
        fraction=train_frac,
        shuffle=True,
        seed=42,
    )

    # CIFAR10 test set (eval transforms)
    test_set = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=eval_transform,
    )

    num_classes = 10

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, num_classes

def _split_subset(dataset, fractions, seed=1234):
    """
    Split a dataset (e.g. ImageFolder) into len(fractions) Subsets,
    using deterministic shuffling.
    fractions must sum to 1.0.
    Example: fractions=[0.5,0.5] -> return two halves.
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
        splits.append(Subset(dataset, part))
        start = end
    return splits


def _get_imagenet_subset_splits(
    data_root: str,
    train_frac: float,
    batch_size: int,
    num_workers: int,
    augment: bool,
):
    """
    This version assumes ImageNet-Mini style data at:
        data_root/train/<class>/*.jpg
        data_root/val/<class>/*.jpg
    There is NO explicit test/ directory.
    We'll:
      - load train/ with train transforms (optionally augmented)
      - load val/ with eval transforms
      - split val/ 50/50 into val_subset and test_subset (deterministic)
      - subsample train/ using train_frac
      - return train_loader, val_loader, test_loader, num_classes
    """

    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")

    train_transform = build_train_transforms_imagenet() if augment else build_eval_transforms()
    eval_transform  = build_eval_transforms()

    full_train_ds = datasets.ImageFolder(
        train_dir,
        transform=train_transform,
    )

    full_val_ds_evalview = datasets.ImageFolder(
        val_dir,
        transform=eval_transform,
    )

    num_classes = len(full_train_ds.classes)

    # deterministic split of original 'val' into our val + test
    val_subset, test_subset = _split_subset(
        full_val_ds_evalview,
        fractions=[0.5, 0.5],
        seed=1234,
    )

    # subsample the training set to match train_frac (0.01, 0.1, 1.0, etc.)
    train_ds_sub = _subset_by_fraction(
        full_train_ds,
        fraction=train_frac,
        shuffle=True,
        seed=42,
    )

    # build loaders
    train_loader = DataLoader(
        train_ds_sub,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, num_classes

def get_dataloaders(
    dataset_name: Literal["cifar10", "imagenet_subset"],
    data_root: str,
    train_frac: float,
    batch_size: int,
    num_workers: int = 4,
    augment: bool = False,
) -> Dict:
    """
    Unified public function.
    Returns:
        {
          "train": train_loader,
          "val": val_loader,
          "test": test_loader,
          "num_classes": num_classes
        }
    """

    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        train_loader, val_loader, test_loader, num_classes = _get_cifar10_splits(
            data_root=data_root,
            train_frac=train_frac,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
        )
    elif dataset_name in ["imagenet_subset", "imagenet-subset", "imagenet"]:
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
