# src/models.py

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple

# at top (you already have this, just confirm)
# from .crossfreq_vit import CrossFreqViT



def build_resnet152(num_classes: int) -> nn.Module:
    """
    Load ResNet-152 pretrained on ImageNet-1k and replace the final FC layer.
    """
    # weights arg name changed in newer torchvision; using the new API
    weights = models.ResNet152_Weights.IMAGENET1K_V1
    model = models.resnet152(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def build_vit_b16(num_classes: int) -> nn.Module:
    """
    Load ViT-B/16 pretrained on ImageNet-1k and replace the classification head.
    """
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    return model

# def build_vit_crossfreq(
#     num_classes: int,
#     fusion_at: int = 6,
#     lf_tokens: int = 4,
#     lf_cutoff: float = 0.15,
#     hf_bins: int = 16,
#     hf_cutoff: float = 0.15,
#     pretrained: bool = True,
# ) -> nn.Module:
#     return CrossFreqViT(
#         num_classes=num_classes,
#         backbone="vit_base_patch16_224",
#         pretrained=pretrained,
#         fusion_at=fusion_at,
#         lf_tokens=lf_tokens,
#         lf_cutoff=lf_cutoff,
#         hf_bins=hf_bins,
#         hf_cutoff=hf_cutoff,
#         num_heads=8,
#         drop=0.0,
#     )


def maybe_freeze_backbone(model: nn.Module, freeze_backbone: bool, model_name: str):
    """
    Optionally freeze everything except the final classification head.
    We'll keep this hook for future ablations on 'low data'.
    By default we won't freeze (freeze_backbone=False).
    """
    if not freeze_backbone:
        return model

    # For ResNet, final head is model.fc
    # For ViT, final head is model.heads.head
    if "resnet" in model_name.lower():
        for name, param in model.named_parameters():
            if name.startswith("fc."):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif "vit" in model_name.lower():
        for name, param in model.named_parameters():
            if name.startswith("heads.head"):
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        # default: freeze nothing if we don't recognize name
        pass

    return model


# def get_model(
#     model_name: str,
#     num_classes: int,
#     device: str,
#     freeze_backbone: bool = False,
# ) -> Tuple[nn.Module, str]:
#     """
#     model_name: "resnet152" or "vit_b16"
#     num_classes: number of classes for the dataset (e.g. 10 for CIFAR-10)
#     device: "cuda" or "cpu"
#     freeze_backbone: set True if you want linear-probing style training
#     returns: (model, model_id_string_for_logging)
#     """

#     model_name = model_name.lower()

#     if model_name in ["resnet152", "resnet-152"]:
#         model = build_resnet152(num_classes)
#         model_id = "resnet152_imagenet_pretrained"
#     elif model_name in ["vit_b16", "vit-b16", "vit_base", "vit-b/16", "vit_base_patch16"]:
#         model = build_vit_b16(num_classes)
#         model_id = "vit_b16_imagenet_pretrained"
#     else:
#         raise ValueError(f"Unknown model '{model_name}'. Use 'resnet152' or 'vit_b16'.")

#     model = maybe_freeze_backbone(model, freeze_backbone, model_name)
#     model = model.to(device)

#     return model, model_id


# change the signature to accept model_kwargs (default None)
def get_model(
    model_name: str,
    num_classes: int,
    device: str,
    freeze_backbone: bool = False,
    model_kwargs: dict | None = None,   # <— add this
) -> Tuple[nn.Module, str]:
    model_kwargs = model_kwargs or {}
    model_name = model_name.lower()

    if model_name in ["resnet152", "resnet-152"]:
        model = build_resnet152(num_classes)
        model_id = "resnet152_imagenet_pretrained"

    elif model_name in ["vit_b16", "vit-b16", "vit_base", "vit-b/16", "vit_base_patch16"]:
        model = build_vit_b16(num_classes)
        model_id = "vit_b16_imagenet_pretrained"

    elif model_name in ["vit_crossfreq", "vit-crossfreq"]:  # <— NEW
        model = CrossFreqViT(num_classes=num_classes, **model_kwargs)
        model_id = "vit_crossfreq"

    else:
        raise ValueError(f"Unknown model '{model_name}'. Use 'resnet152', 'vit_b16', or 'vit_crossfreq'.")

    model = maybe_freeze_backbone(model, freeze_backbone, model_name)
    model = model.to(device)
    return model, model_id
