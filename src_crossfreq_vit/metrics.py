# src_crossfreq_vit/metrics.py
# Evaluation metrics and confusion-matrix utilities for CrossFreq-ViT.

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import itertools
from typing import Dict, Any, Tuple, Optional


@torch.no_grad()
def collect_logits_and_labels(model, dataloader, device: str):
    """
    Run the model on the full dataloader and collect logits and labels.
    Returns:
        logits: [N, C] tensor
        labels: [N] tensor
    """
    model.eval()
    all_logits, all_labels = [], []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)  # (B,C)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_logits, all_labels


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy in %."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return 100.0 * correct / labels.size(0)


def compute_precision_recall_f1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[list] = None,
):
    """
    Returns:
        per_class: list of dicts {class, precision, recall, f1}
        macro: dict {precision_macro, recall_macro, f1_macro}
    """
    preds = torch.argmax(logits, dim=1).numpy()
    y_true = labels.numpy()

    # Per-class
    prec_pc, rec_pc, f1_pc, _ = precision_recall_fscore_support(
        y_true, preds, average=None, zero_division=0
    )
    # Macro average
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, preds, average="macro", zero_division=0
    )

    per_class = []
    for i in range(len(prec_pc)):
        cname = class_names[i] if class_names and i < len(class_names) else str(i)
        per_class.append({
            "class": cname,
            "precision": float(prec_pc[i] * 100.0),
            "recall": float(rec_pc[i] * 100.0),
            "f1": float(f1_pc[i] * 100.0),
        })

    macro = {
        "precision_macro": float(prec_m * 100.0),
        "recall_macro": float(rec_m * 100.0),
        "f1_macro": float(f1_m * 100.0),
    }

    return per_class, macro


def plot_confusion_matrix(cm, class_names=None, normalized=True):
    """
    Create a heatmap for the confusion matrix.
    """
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
    )
    ax.set_ylabel("True label", labelpad=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text = f"{cm[i, j]*100.0:.1f}%" if normalized else f"{cm[i, j]:.0f}"
        ax.text(j, i, text, ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=9)

    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def compute_confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[list] = None,
    normalize: bool = True,
):
    """
    Build confusion matrix. If normalize==True, rows sum to 1.
    Returns (cm_array, fig)
    """
    preds = torch.argmax(logits, dim=1).numpy()
    y_true = labels.numpy()

    cm_raw = confusion_matrix(y_true, preds)
    cm = cm_raw.astype(np.float32)
    if normalize:
        cm /= (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig = plot_confusion_matrix(cm, class_names, normalized=normalize)
    return cm, fig


def evaluate_full(model, dataloader, device: str, class_names: Optional[list] = None) -> Dict[str, Any]:
    """
    Full evaluation on TEST set after training.
    Returns dict:
        {
          "accuracy": float,
          "per_class_metrics": [...],
          "macro_metrics": {...},
          "confusion_matrix": ndarray,
          "confusion_matrix_fig": matplotlib.figure.Figure
        }
    """
    logits, labels = collect_logits_and_labels(model, dataloader, device)
    acc = compute_accuracy(logits, labels)
    per_class, macro = compute_precision_recall_f1(logits, labels, class_names)
    cm, cm_fig = compute_confusion_matrix(logits, labels, class_names, normalize=True)

    return {
        "accuracy": acc,
        "per_class_metrics": per_class,
        "macro_metrics": macro,
        "confusion_matrix": cm,
        "confusion_matrix_fig": cm_fig,
    }
