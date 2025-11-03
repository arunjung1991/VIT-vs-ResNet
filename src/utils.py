# src/utils.py

import os
import random
import torch
import wandb
import math
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda:1" if torch.cuda.is_available() else "cpu"


def init_wandb(project: str,
               model_name: str,
               dataset_name: str,
               train_frac: float,
               config_extra: dict = None):
    run_name = f"{model_name}_{dataset_name}_{train_frac}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    config = {
        "model": model_name,
        "dataset": dataset_name,
        "train_frac": train_frac,
    }
    if config_extra:
        config.update(config_extra)

    # ✅ Your wandb initialization with entity included
    wandb.init(
        entity="arunjung1991",   # 👈 Your wandb username or team
        project=project,
        name=run_name,
        config=config,
    )
    return run_name


def build_optimizer(model, lr=5e-5, weight_decay=0.05):
    """
    AdamW is usually standard for ViTs and fine-tuning ResNets too.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    return optimizer


def build_cosine_scheduler(optimizer, num_warmup_steps, num_total_steps):
    """
    Returns a cosine decay schedule with warmup.
    We'll step this scheduler every iteration (not every epoch).
    """

    def lr_lambda(current_step: int):
        # warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # cosine after warmup
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_total_steps - num_warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


def save_checkpoint(model, path: str):
    """
    Save model weights only.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: str, map_location=None):
    """
    Load model weights only.
    """
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt)
    return model
