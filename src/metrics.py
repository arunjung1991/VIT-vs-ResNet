# src/metrics.py

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import itertools
from typing import Dict, Any, Tuple, Optional


@torch.no_grad()
def collect_logits_and_labels(model, dataloader, device: str):
    """
    Run the model on an entire dataloader and collect:
    - logits [N, C]
    - labels [N]
    Returns (logits_tensor, labels_tensor)
    """
    model.eval()

    all_logits = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)  # [B, C]

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)  # [N, C]
    all_labels = torch.cat(all_labels, dim=0)  # [N]
    return all_logits, all_labels


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    top-1 accuracy in %
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return 100.0 * correct / total


def compute_precision_recall_f1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[list] = None,
):
    """
    Returns:
        per_class: dict with precision/recall/f1 per class index
        macro: dict with macro-avg precision/recall/f1
    """
    preds = torch.argmax(logits, dim=1).numpy()
    y_true = labels.numpy()

    # average=None gives per-class metrics
    prec_pc, rec_pc, f1_pc, _ = precision_recall_fscore_support(
        y_true, preds, average=None, zero_division=0
    )

    # macro averages across classes
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, preds, average="macro", zero_division=0
    )

    per_class = []
    num_classes = len(prec_pc)
    for i in range(num_classes):
        cname = class_names[i] if (class_names and i < len(class_names)) else str(i)
        per_class.append({
            "class": cname,
            "precision": float(prec_pc[i] * 100.0),
            "recall":    float(rec_pc[i] * 100.0),
            "f1":        float(f1_pc[i] * 100.0),
        })

    macro = {
        "precision_macro": float(prec_macro * 100.0),
        "recall_macro":    float(rec_macro * 100.0),
        "f1_macro":        float(f1_macro * 100.0),
    }

    return per_class, macro


def compute_confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[list] = None,
    normalize: bool = True,
):
    """
    Build confusion matrix.
    If normalize==True, rows sum to 1 so you can see misclassification ratios.
    Returns (cm, fig)
        cm: np.ndarray [C, C]
        fig: matplotlib Figure
    """
    preds = torch.argmax(logits, dim=1).numpy()
    y_true = labels.numpy()

    cm_raw = confusion_matrix(y_true, preds)
    cm = cm_raw.astype(np.float32)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-9
        cm = cm / row_sums  # each row = distribution of predictions for that true class

    fig = plot_confusion_matrix(cm, class_names, normalized=normalize)
    return cm, fig


def plot_confusion_matrix(cm, class_names=None, normalized=True):
    """
    Make a matplotlib heatmap for confusion matrix.
    If normalized=True, values are fractions [0,1]. We'll annotate as percentages.
    """
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)

    # Instead of using labelpad inside ax.set(...),
    # we will call set_ylabel separately for compatibility.
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
    )

    # Manually adjust the label padding to avoid the error
    ax.set_ylabel("True label", labelpad=10)

    # Rotate x tick labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # annotate cells
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalized:
            text_val = f"{cm[i, j]*100.0:.1f}%"
        else:
            text_val = f"{cm[i, j]:.0f}"
        ax.text(
            j, i,
            text_val,
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=9,
        )

    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def evaluate_full(
    model,
    dataloader,
    device: str,
    class_names: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Main evaluation on TEST set after training.

    Returns:
    {
      "accuracy": float,
      "per_class_metrics": [ {class, precision, recall, f1}, ... ],
      "macro_metrics": {precision_macro, recall_macro, f1_macro},
      "confusion_matrix": cm (ndarray),
      "confusion_matrix_fig": fig
    }
    """
    logits, labels = collect_logits_and_labels(model, dataloader, device)

    acc = compute_accuracy(logits, labels)

    per_class, macro = compute_precision_recall_f1(
        logits,
        labels,
        class_names=class_names,
    )

    cm, cm_fig = compute_confusion_matrix(
        logits,
        labels,
        class_names=class_names,
        normalize=True,   # show % distribution of predictions per true class
    )

    results = {
        "accuracy": acc,
        "per_class_metrics": per_class,
        "macro_metrics": macro,
        "confusion_matrix": cm,
        "confusion_matrix_fig": cm_fig,
    }

    return results
