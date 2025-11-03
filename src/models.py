# src/models.py

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


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


def get_model(
    model_name: str,
    num_classes: int,
    device: str,
    freeze_backbone: bool = False,
) -> Tuple[nn.Module, str]:
    """
    model_name: "resnet152" or "vit_b16"
    num_classes: number of classes for the dataset (e.g. 10 for CIFAR-10)
    device: "cuda" or "cpu"
    freeze_backbone: set True if you want linear-probing style training
    returns: (model, model_id_string_for_logging)
    """

    model_name = model_name.lower()

    if model_name in ["resnet152", "resnet-152"]:
        model = build_resnet152(num_classes)
        model_id = "resnet152_imagenet_pretrained"
    elif model_name in ["vit_b16", "vit-b16", "vit_base", "vit-b/16", "vit_base_patch16"]:
        model = build_vit_b16(num_classes)
        model_id = "vit_b16_imagenet_pretrained"
    else:
        raise ValueError(f"Unknown model '{model_name}'. Use 'resnet152' or 'vit_b16'.")

    model = maybe_freeze_backbone(model, freeze_backbone, model_name)
    model = model.to(device)

    return model, model_id
