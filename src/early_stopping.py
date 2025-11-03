# src/early_stopping.py

import math
from typing import Optional
from .utils import save_checkpoint


class EarlyStopping:
    """
    Monitors a validation metric and triggers early stop if it doesn't improve.

    Typical usage in training loop:
    --------------------------------
    stopper = EarlyStopping(
        patience=5,
        checkpoint_path="checkpoints/best_model.pth",
        mode="min",               # "min" for val_loss, "max" for val_acc
    )

    for epoch in range(max_epochs):
        train(...)
        val_metric = validate(...)

        stop, best_path = stopper.step(val_metric, model)
        if stop:
            break

    After training, best_path is the file we should load for final test evaluation.
    """

    def __init__(
        self,
        patience: int,
        checkpoint_path: str,
        mode: str = "min",
        min_delta: float = 0.0,
    ):
        """
        patience:    how many epochs we wait with no improvement
        checkpoint_path: where to save the best model weights
        mode: "min"  -> we expect metric to go DOWN (e.g. val_loss)
              "max"  -> we expect metric to go UP   (e.g. val_acc)
        min_delta:   minimum change to count as an improvement

        Internals:
        - best_score: best metric observed so far
        - bad_epochs: how many epochs since last improvement
        - should_stop: becomes True when patience is exceeded
        - best_path: path to the best weights on disk
        """
        assert mode in ["min", "max"], "mode must be 'min' or 'max'"
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self.min_delta = min_delta

        if mode == "min":
            self.best_score = math.inf
        else:
            self.best_score = -math.inf

        self.bad_epochs = 0
        self.should_stop = False
        self.best_path: Optional[str] = None

    def _is_improved(self, current: float) -> bool:
        if self.mode == "min":
            # want current < best - min_delta
            return (self.best_score - current) > self.min_delta
        else:
            # mode == "max"
            return (current - self.best_score) > self.min_delta

    def step(self, current_score: float, model) -> (bool, Optional[str]):
        """
        Call this at the end of each epoch with the current validation metric
        (e.g., val_loss if mode=="min", val_acc if mode=="max").

        If improved:
            - save checkpoint
            - reset bad_epochs
        Else:
            - bad_epochs += 1
            - if bad_epochs > patience: should_stop = True

        Returns:
            (should_stop: bool, best_path: Optional[str])
        """

        if self._is_improved(current_score):
            # Update best
            self.best_score = current_score
            self.bad_epochs = 0
            save_checkpoint(model, self.checkpoint_path)
            self.best_path = self.checkpoint_path
        else:
            # No improvement
            self.bad_epochs += 1
            if self.bad_epochs > self.patience:
                self.should_stop = True

        return self.should_stop, self.best_path
