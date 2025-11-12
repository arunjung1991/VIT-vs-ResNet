# src_crossfreq_vit/early_stopping.py
# Simple early stopping with checkpoint saving.

import math
from typing import Optional
from .utils_crossfreq_vit import save_checkpoint


class EarlyStopping:
    """
    Monitors a validation metric and stops training when it stops improving.

    Example:
    --------
        stopper = EarlyStopping(
            patience=5,
            checkpoint_path="checkpoints/best_model.pth",
            mode="min",               # "min" for val_loss, "max" for val_acc
        )

        for epoch in range(max_epochs):
            ...
            val_loss = validate(...)
            stop, best_path = stopper.step(val_loss, model)
            if stop:
                break
    """

    def __init__(
        self,
        patience: int,
        checkpoint_path: str,
        mode: str = "min",
        min_delta: float = 0.0,
    ):
        assert mode in ["min", "max"], "mode must be 'min' or 'max'"
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self.min_delta = min_delta

        # initialize depending on direction
        self.best_score = math.inf if mode == "min" else -math.inf
        self.bad_epochs = 0
        self.should_stop = False
        self.best_path: Optional[str] = None

    # ---- internal helper ----
    def _is_improved(self, current: float) -> bool:
        if self.mode == "min":
            return (self.best_score - current) > self.min_delta
        else:
            return (current - self.best_score) > self.min_delta

    # ---- public ----
    def step(self, current_score: float, model) -> (bool, Optional[str]):
        """
        Evaluate after each epoch. Saves best checkpoint automatically.
        Returns (should_stop, best_path).
        """
        if self._is_improved(current_score):
            self.best_score = current_score
            self.bad_epochs = 0
            save_checkpoint(model, self.checkpoint_path)
            self.best_path = self.checkpoint_path
        else:
            self.bad_epochs += 1
            if self.bad_epochs > self.patience:
                self.should_stop = True
        return self.should_stop, self.best_path
