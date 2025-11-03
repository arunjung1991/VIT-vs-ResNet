# src/train_loop.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Tuple
import numpy as np
import wandb
from tqdm import tqdm


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    logits: [B, C]
    targets: [B] int64 class ids
    returns percentage accuracy as float
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def apply_mixup(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    device: str,
):
    """
    Classic Mixup.
    Returns:
      mixed_images,
      (y_a, y_b, lam) for computing mixup loss,
      hard_targets_for_logging_accuracy
    If alpha <= 0, we skip and behave like normal supervised training.
    """
    if alpha is None or alpha <= 0.0:
        return images, (targets, None, None), targets

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=device)

    mixed_x = lam * images + (1.0 - lam) * images[index, :]

    y_a = targets
    y_b = targets[index]

    # for accuracy logging, we just keep y_a as "the" label
    return mixed_x, (y_a, y_b, lam), y_a


def mixup_criterion(
    criterion,
    logits,
    y_a,
    y_b,
    lam,
):
    """
    If we used mixup, compute lam * CE(logits, y_a) + (1-lam) * CE(logits, y_b).
    If we didn't use mixup, lam will be None and we just do standard CE.
    """
    if lam is None or y_b is None:
        return criterion(logits, y_a)
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    mixup_alpha: float,
) -> Tuple[float, float]:
    """
    One epoch of training.
    Steps LR scheduler *per batch* (cosine w/ warmup style).
    Returns (avg_train_loss, avg_train_acc_percent)
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    pbar = tqdm(dataloader, desc="train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup (optional)
        images, (y_a, y_b, lam), acc_targets = apply_mixup(
            images, labels, alpha=mixup_alpha, device=device
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)  # [B, num_classes]

        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        loss.backward()
        optimizer.step()

        # step the LR scheduler per batch
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            # compute accuracy using acc_targets (usually y_a)
            batch_acc = accuracy_from_logits(logits, acc_targets)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc += batch_acc * bs
        total_count += bs

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{batch_acc:.2f}%",
        })

    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count
    return avg_loss, avg_acc


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Tuple[float, float]:
    """
    Validation / evaluation pass.
    No mixup, no grad.
    Returns (avg_val_loss, avg_val_acc_percent)
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    pbar = tqdm(dataloader, desc="val", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_acc = accuracy_from_logits(logits, labels)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc += batch_acc * bs
        total_count += bs

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{batch_acc:.2f}%",
        })

    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count
    return avg_loss, avg_acc


def run_epoch_and_log(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epoch_idx: int,
    mixup_alpha: float,
):
    """
    Convenience wrapper we will call from run_experiment.py each epoch.
    - trains
    - validates
    - logs to wandb
    - returns val_loss, val_acc (for early stopping)
    """

    train_loss, train_acc = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixup_alpha=mixup_alpha,
    )

    val_loss, val_acc = validate_one_epoch(
        model=model,
        dataloader=val_loader,
        device=device,
    )

    # Log to wandb
    wandb.log({
        "epoch": epoch_idx,
        "train/loss": train_loss,
        "train/acc": train_acc,
        "val/loss": val_loss,
        "val/acc": val_acc,
        "lr": optimizer.param_groups[0]["lr"],
    })

    return val_loss, val_acc
