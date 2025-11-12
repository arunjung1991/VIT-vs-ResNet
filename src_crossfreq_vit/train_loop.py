# src_crossfreq_vit/train_loop.py
# Generic training and validation loops with optional ArcFace support.

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Tuple
import numpy as np
import wandb
from tqdm import tqdm


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return 100.0 * correct / targets.size(0)


# ---------------------------------------------------------------------
# MixUp utilities
# ---------------------------------------------------------------------

def apply_mixup(images: torch.Tensor, targets: torch.Tensor, alpha: float, device: str):
    """
    Standard MixUp augmentation. If alpha <= 0, returns inputs unchanged.
    """
    if alpha is None or alpha <= 0.0:
        return images, (targets, None, None), targets

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=device)

    mixed_x = lam * images + (1.0 - lam) * images[index, :]
    y_a = targets
    y_b = targets[index]

    return mixed_x, (y_a, y_b, lam), y_a


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    if lam is None or y_b is None:
        return criterion(logits, y_a)
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


# ---------------------------------------------------------------------
# Train / validate epochs
# ---------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    mixup_alpha: float,
    *,
    arc_head: Optional[torch.nn.Module] = None,
    use_features_fn: bool = False,
) -> Tuple[float, float]:
    """
    One training epoch.
    If arc_head is None → standard CE training.
    If arc_head is provided → ArcFace mode (no MixUp or label smoothing).
    """
    model.train()
    is_arcface = arc_head is not None
    total_loss, total_acc, total_count = 0.0, 0.0, 0

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    if is_arcface:
        criterion = torch.nn.CrossEntropyLoss()
        mixup_alpha = 0.0  # disable mixup in ArcFace mode

    pbar = tqdm(dataloader, desc="train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        images, (y_a, y_b, lam), acc_targets = apply_mixup(images, labels, alpha=mixup_alpha, device=device)

        optimizer.zero_grad(set_to_none=True)

        if is_arcface:
            feats = model.forward_features(images) if use_features_fn else model(images)
            logits = arc_head(feats, labels)
            loss = criterion(logits, labels)
            batch_acc = accuracy_from_logits(logits.detach(), labels)
        else:
            logits = model(images)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            batch_acc = accuracy_from_logits(logits.detach(), acc_targets)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc += batch_acc * bs
        total_count += bs
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.2f}%"})

    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count
    return avg_loss, avg_acc


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    *,
    arc_head: Optional[torch.nn.Module] = None,
    use_features_fn: bool = False,
) -> Tuple[float, float]:
    """
    Validation / evaluation loop.
    If arc_head is given → ArcFace eval mode (no margin).
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    is_arcface = arc_head is not None

    total_loss, total_acc, total_count = 0.0, 0.0, 0
    pbar = tqdm(dataloader, desc="val", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        if is_arcface:
            feats = model.forward_features(images) if use_features_fn else model(images)
            logits = arc_head(feats, eval_mode=True)
        else:
            logits = model(images)

        loss = criterion(logits, labels)
        batch_acc = accuracy_from_logits(logits, labels)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc += batch_acc * bs
        total_count += bs
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.2f}%"})

    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count
    return avg_loss, avg_acc


# ---------------------------------------------------------------------
# Unified per-epoch wrapper with wandb logging
# ---------------------------------------------------------------------

def run_epoch_and_log(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epoch_idx: int,
    mixup_alpha: float,
    *,
    arc_head: Optional[torch.nn.Module] = None,
    use_features_fn: bool = False,
):
    """
    Combined train + val step with wandb logging.
    """
    train_loss, train_acc = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixup_alpha=mixup_alpha,
        arc_head=arc_head,
        use_features_fn=use_features_fn,
    )

    val_loss, val_acc = validate_one_epoch(
        model=model,
        dataloader=val_loader,
        device=device,
        arc_head=arc_head,
        use_features_fn=use_features_fn,
    )

    wandb.log({
        "epoch": epoch_idx,
        "train/loss": train_loss,
        "train/acc": train_acc,
        "val/loss": val_loss,
        "val/acc": val_acc,
        "lr": optimizer.param_groups[0]["lr"],
    })

    return val_loss, val_acc
