# src_crossfreq_vit/model_crossfreq_vit.py
# Factory helpers to build and configure CrossFreq-ViT.

from __future__ import annotations
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn

from .crossfreq_vit import CrossFreqViT


__all__ = [
    "build_vit_crossfreq",
    "maybe_freeze_crossfreq_backbone",
    "get_crossfreq_model",
]


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------

def build_vit_crossfreq(
    num_classes: int,
    *,
    pretrained: bool = True,
    fusion_at: int = 6,
    lf_tokens: int = 4,
    lf_cutoff: float = 0.15,
    hf_bins: int = 16,
    hf_cutoff: float = 0.15,
    num_heads: int = 8,
    drop: float = 0.0,
) -> nn.Module:
    """
    Factory for CrossFreqViT with consistent defaults.
    """
    return CrossFreqViT(
        num_classes=num_classes,
        backbone="vit_base_patch16_224",
        pretrained=pretrained,
        fusion_at=fusion_at,
        lf_tokens=lf_tokens,
        lf_cutoff=lf_cutoff,
        hf_bins=hf_bins,
        hf_cutoff=hf_cutoff,
        num_heads=num_heads,
        drop=drop,
    )


# ---------------------------------------------------------------------
# Freeze utility
# ---------------------------------------------------------------------

def maybe_freeze_crossfreq_backbone(model: nn.Module, freeze_backbone: bool) -> nn.Module:
    """
    Optionally freeze everything except the final classifier.
    """
    if not freeze_backbone:
        return model

    for name, p in model.named_parameters():
        if name.startswith("classifier."):
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


# ---------------------------------------------------------------------
# High-level getter
# ---------------------------------------------------------------------

def get_crossfreq_model(
    *,
    num_classes: int,
    device: str = "cuda",
    freeze_backbone: bool = False,
    model_kwargs: Optional[Dict] = None,
) -> Tuple[nn.Module, str]:
    """
    Convenience wrapper returning (model, model_id).

    Example:
    --------
        model, mid = get_crossfreq_model(
            num_classes=1000,
            device="cuda",
            freeze_backbone=False,
            model_kwargs=dict(pretrained=True, fusion_at=6, lf_tokens=4, hf_bins=16),
        )
    """
    mk = dict(
        pretrained=True,
        fusion_at=6,
        lf_tokens=4,
        lf_cutoff=0.15,
        hf_bins=16,
        hf_cutoff=0.15,
        num_heads=8,
        drop=0.0,
    )
    if model_kwargs:
        mk.update(model_kwargs)

    model = build_vit_crossfreq(num_classes=num_classes, **mk)
    model = maybe_freeze_crossfreq_backbone(model, freeze_backbone)
    model = model.to(device)
    return model, "vit_crossfreq"
