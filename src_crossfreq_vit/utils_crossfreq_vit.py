# src_crossfreq_vit/utils_crossfreq_vit.py
# Utility functions for training CrossFreq-ViT.

import os
import math
import random
import time
from typing import Any, Dict, Iterable, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import wandb


# ---------------------------------------------------------------------
# Reproducibility and device
# ---------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # allow perf autotune
    torch.backends.cudnn.benchmark = True


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------

def init_wandb(
    project: str,
    model_name: str,
    dataset_name: str,
    train_frac: float,
    config_extra: Dict[str, Any] | None = None,
) -> str:
    """
    Initialize a wandb run and return a run_name string.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{model_name}_{dataset_name}_{train_frac}_{ts}"
    cfg = dict(model=model_name, dataset=dataset_name, train_frac=train_frac, run_name=run_name)
    if config_extra:
        cfg.update(config_extra)
    wandb.init(project=project, name=run_name, config=cfg)
    return run_name


# ---------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------

def build_optimizer(model_or_params: Union[nn.Module, Iterable], lr: float, weight_decay: float) -> AdamW:
    """
    Accept either a model or explicit params iterable.
    """
    if hasattr(model_or_params, "parameters"):
        params = model_or_params.parameters()
    else:
        params = model_or_params
    return AdamW(params, lr=lr, weight_decay=weight_decay)


# ---------------------------------------------------------------------
# Cosine LR with warmup
# ---------------------------------------------------------------------

def _cosine_with_warmup_lambda(num_warmup_steps: int, num_total_steps: int):
    num_warmup_steps = max(0, int(num_warmup_steps))
    num_total_steps = max(1, int(num_total_steps))

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_total_steps - num_warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def build_cosine_scheduler(optimizer: AdamW, num_warmup_steps: int, num_total_steps: int) -> LambdaLR:
    lam = _cosine_with_warmup_lambda(num_warmup_steps, num_total_steps)
    return LambdaLR(optimizer, lr_lambda=lam)


# ---------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------

def save_checkpoint(model: nn.Module, path: str) -> None:
    """Save model weights."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def save_arcface_ckpt(path: str, backbone: nn.Module, arc_head: nn.Module, extra: Dict[str, Any] | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"backbone": backbone.state_dict(), "arc_head": arc_head.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_arcface_ckpt(
    path: str,
    backbone: nn.Module,
    arc_head: nn.Module,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    if "backbone" in ckpt:
        backbone.load_state_dict(ckpt["backbone"], strict=strict)
    if "arc_head" in ckpt:
        arc_head.load_state_dict(ckpt["arc_head"], strict=strict)
    return ckpt
