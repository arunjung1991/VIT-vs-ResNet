# src/eval.py

import torch
import wandb
from typing import Optional, Dict, Any, List
from .utils import load_checkpoint
from .metrics import evaluate_full


def evaluate_and_log(
    model: torch.nn.Module,
    checkpoint_path: str,
    test_loader,
    device: str,
    class_names: Optional[List[str]] = None,
):
    """
    1. Load best checkpoint weights into `model`.
    2. Run full evaluation on the test set.
    3. Log metrics + confusion matrix figure to wandb.
    4. Return the results dict.
    """

    # Load best weights before final eval
    model = load_checkpoint(model, checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    results: Dict[str, Any] = evaluate_full(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
    )

    # Log scalar metrics to wandb
    wandb.log({
        "test/accuracy": results["accuracy"],
        "test/precision_macro": results["macro_metrics"]["precision_macro"],
        "test/recall_macro": results["macro_metrics"]["recall_macro"],
        "test/f1_macro": results["macro_metrics"]["f1_macro"],
    })

    # Log per-class metrics as a table-like structure
    # (wandb will show this in the UI nicely)
    # We'll prefix keys with test/ so it's obvious
    for class_metric in results["per_class_metrics"]:
        cname = class_metric["class"]
        wandb.log({
            f"test/{cname}_precision": class_metric["precision"],
            f"test/{cname}_recall":    class_metric["recall"],
            f"test/{cname}_f1":        class_metric["f1"],
        })

    # Log confusion matrix figure to wandb
    cm_fig = results["confusion_matrix_fig"]
    wandb.log({
        "test/confusion_matrix_fig": wandb.Image(cm_fig)
    })

    return results
