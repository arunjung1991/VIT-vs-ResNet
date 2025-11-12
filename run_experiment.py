# run_experiment.py

import argparse
import math
import os
import torch
import wandb

from src.utils import (
    set_seed,
    get_device,
    init_wandb,
    build_optimizer,
    build_cosine_scheduler,
)
from src.datasets import get_dataloaders
from src.models import get_model
from src.train_loop import run_epoch_and_log
from src.early_stopping import EarlyStopping
from src.eval import evaluate_and_log


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ResNet152 vs ViT-B/16 on CIFAR10 / ImageNet subset with data scaling"
    )

    # dataset / model choices
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        choices=["cifar10", "imagenet_subset"],
                        help="Which dataset to use.")
    parser.add_argument("--data_root",
                        type=str,
                        required=True,
                        help="Path to dataset root. CIFAR10 will download here.")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        choices=["resnet152", "vit_b16"],
                        help="Model architecture.")

    # experiment knobs
    parser.add_argument("--train_frac",
                        type=float,
                        default=1.0,
                        help="Fraction of training data to use (0<frac<=1).")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128)
    parser.add_argument("--epochs",
                        type=int,
                        default=50)
    parser.add_argument("--seed",
                        type=int,
                        default=42)

    parser.add_argument("--augment",
                        type=str,
                        default="True",
                        help="Use data augmentation? True/False (affects train transforms)")

    parser.add_argument("--mixup_alpha",
                        type=float,
                        default=0.2,
                        help="Mixup strength. 0.0 disables mixup.")

    # optimization settings
    parser.add_argument("--lr",
                        type=float,
                        default=5e-5,
                        help="Base learning rate for AdamW.")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.05,
                        help="AdamW weight decay.")
    parser.add_argument("--warmup_steps",
                        type=int,
                        default=500,
                        help="Number of warmup steps for cosine LR.")

    # early stopping + checkpoints
    parser.add_argument("--patience",
                        type=int,
                        default=5,
                        help="Early stopping patience (epochs without improvement).")
    parser.add_argument("--checkpoint_dir",
                        type=str,
                        default="checkpoints",
                        help="Where to save best model weights.")

    # wandb
    parser.add_argument("--wandb_project",
                        type=str,
                        default="vit_vs_resnet",
                        help="wandb project name")

    return parser.parse_args()


def str2bool(v: str) -> bool:
    return v.lower() in ["1", "true", "yes", "y"]


def main():
    args = parse_args()

    # prep
    set_seed(args.seed)
    device = get_device()
    print(f"[INFO] Device: {device}")

    use_augment = str2bool(args.augment)

    # Build dataloaders
    # this returns dict with train/val/test/num_classes
    loaders = get_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        train_frac=args.train_frac,
        batch_size=args.batch_size,
        num_workers=4,
        augment=use_augment,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    num_classes = loaders["num_classes"]

    # Build model
    model, model_id = get_model(
        model_name=args.model,
        num_classes=num_classes,
        device=device,
        freeze_backbone=False,  # full finetune by default
    )

    # Init wandb
    run_name = init_wandb(
        project=args.wandb_project,
        model_name=model_id,
        dataset_name=args.dataset,
        train_frac=args.train_frac,
        config_extra=dict(
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            patience=args.patience,
            mixup_alpha=args.mixup_alpha,
            augment=use_augment,
            seed=args.seed,
        ),
    )

    # Optimizer
    optimizer = build_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    # we are doing cosine decay with warmup stepped per batch
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(args.warmup_steps, total_steps // 10 if total_steps > 10 else args.warmup_steps)
    scheduler = build_cosine_scheduler(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_total_steps=total_steps,
    )

    # Early stopping
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{run_name}.pth")

    stopper = EarlyStopping(
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        mode="min",       # monitor val_loss (lower is better)
        min_delta=0.0,
    )

    print(f"[INFO] Starting training for up to {args.epochs} epochs...")
    print(f"[INFO] Best checkpoint will be saved to: {checkpoint_path}")

    best_path = None
    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")

        val_loss, val_acc = run_epoch_and_log(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch_idx=epoch,
            mixup_alpha=args.mixup_alpha,
        )

        print(f"[Epoch {epoch+1}] val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")

        # step early stopping using val_loss
        should_stop, best_path = stopper.step(val_loss, model)

        if should_stop:
            print("[INFO] Early stopping triggered.")
            break

    # Safety: if best_path is still None (no improvement?), fall back to last state
    if best_path is None:
        best_path = checkpoint_path
        torch.save(model.state_dict(), best_path)

    print(f"[INFO] Best model path: {best_path}")

    # Final test evaluation + wandb logging
    print("[INFO] Running final evaluation on test set...")
    _ = evaluate_and_log(
        model=model,
        checkpoint_path=best_path,
        test_loader=test_loader,
        device=device,
        class_names=None,   # you can pass class names list later if you want pretty labels
    )

    print("[INFO] Done.")
    wandb.finish()


if __name__ == "__main__":
    main()

# # run_experiment.py

# import argparse
# import math
# import os
# import torch
# import wandb

# from src.utils import (
#     set_seed,
#     get_device,
#     init_wandb,
#     build_optimizer,
#     build_cosine_scheduler,
# )
# from src.datasets import get_dataloaders
# from src.models import get_model
# from src.train_loop import run_epoch_and_log
# from src.early_stopping import EarlyStopping
# from src.eval import evaluate_and_log


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Compare ResNet152, ViT-B/16, and CrossFreqViT on CIFAR10 / ImageNet subset with data scaling"
#     )

#     # dataset / model choices
#     parser.add_argument("--dataset",
#                         type=str,
#                         required=True,
#                         choices=["cifar10", "imagenet_subset"],
#                         help="Which dataset to use.")
#     parser.add_argument("--data_root",
#                         type=str,
#                         required=True,
#                         help="Path to dataset root. CIFAR10 will download here.")
#     parser.add_argument("--model",
#                         type=str,
#                         required=True,
#                         choices=["resnet152", "vit_b16", "vit_crossfreq"],
#                         help="Model architecture.")

#     # experiment knobs
#     parser.add_argument("--train_frac",
#                         type=float,
#                         default=1.0,
#                         help="Fraction of training data to use (0<frac<=1).")
#     parser.add_argument("--batch_size",
#                         type=int,
#                         default=128)
#     parser.add_argument("--epochs",
#                         type=int,
#                         default=50)
#     parser.add_argument("--seed",
#                         type=int,
#                         default=42)

#     parser.add_argument("--augment",
#                         type=str,
#                         default="True",
#                         help="Use data augmentation? True/False (affects train transforms)")

#     parser.add_argument("--mixup_alpha",
#                         type=float,
#                         default=0.2,
#                         help="Mixup strength. 0.0 disables mixup.")

#     # optimization settings
#     parser.add_argument("--lr",
#                         type=float,
#                         default=5e-5,
#                         help="Base learning rate for AdamW.")
#     parser.add_argument("--weight_decay",
#                         type=float,
#                         default=0.05,
#                         help="AdamW weight decay.")
#     parser.add_argument("--warmup_steps",
#                         type=int,
#                         default=500,
#                         help="Number of warmup steps for cosine LR.")

#     # early stopping + checkpoints
#     parser.add_argument("--patience",
#                         type=int,
#                         default=5,
#                         help="Early stopping patience (epochs without improvement).")
#     parser.add_argument("--checkpoint_dir",
#                         type=str,
#                         default="checkpoints",
#                         help="Where to save best model weights.")

#     # wandb
#     parser.add_argument("--wandb_project",
#                         type=str,
#                         default="vit_vs_resnet",
#                         help="wandb project name")

#     # ---- CrossFreqViT specific knobs (ignored by other models) ----
#     parser.add_argument("--fusion_at",
#                         type=int,
#                         default=6,
#                         help="Start of 'second half' (0-based ViT block index to start collecting CLS tokens).")
#     parser.add_argument("--lf_tokens",
#                         type=int,
#                         default=4,
#                         help="Number of global low-frequency tokens (e.g., 4 for 2x2 pooling).")
#     parser.add_argument("--hf_bins",
#                         type=int,
#                         default=16,
#                         help="Per-patch high-frequency descriptor bins (e.g., 16 for 4x4 pooling).")
#     parser.add_argument("--lf_cutoff",
#                         type=float,
#                         default=0.15,
#                         help="Low-pass cutoff ratio (0..1) in frequency domain.")
#     parser.add_argument("--hf_cutoff",
#                         type=float,
#                         default=0.15,
#                         help="High-pass cutoff ratio (0..1) in frequency domain.")

#     return parser.parse_args()


# def str2bool(v: str) -> bool:
#     return v.lower() in ["1", "true", "yes", "y"]


# def main():
#     args = parse_args()

#     # prep
#     set_seed(args.seed)
#     device = get_device()
#     print(f"[INFO] Device: {device}")

#     use_augment = str2bool(args.augment)

#     # Build dataloaders
#     loaders = get_dataloaders(
#         dataset_name=args.dataset,
#         data_root=args.data_root,
#         train_frac=args.train_frac,
#         batch_size=args.batch_size,
#         num_workers=4,
#         augment=use_augment,
#     )

#     train_loader = loaders["train"]
#     val_loader = loaders["val"]
#     test_loader = loaders["test"]
#     num_classes = loaders["num_classes"]

#     # Build model
#     if args.model == "vit_crossfreq":
#         model, model_id = get_model(
#             model_name=args.model,
#             num_classes=num_classes,
#             device=device,
#             freeze_backbone=False,  # full finetune by default
#             model_kwargs=dict(
#                 fusion_at=args.fusion_at,
#                 lf_tokens=args.lf_tokens,
#                 lf_cutoff=args.lf_cutoff,
#                 hf_bins=args.hf_bins,
#                 hf_cutoff=args.hf_cutoff,
#             ),
#         )
#     else:
#         model, model_id = get_model(
#             model_name=args.model,
#             num_classes=num_classes,
#             device=device,
#             freeze_backbone=False,  # full finetune by default
#         )

#     # Init wandb
#     run_name = init_wandb(
#         project=args.wandb_project,
#         model_name=model_id,
#         dataset_name=args.dataset,
#         train_frac=args.train_frac,
#         config_extra=dict(
#             batch_size=args.batch_size,
#             epochs=args.epochs,
#             lr=args.lr,
#             weight_decay=args.weight_decay,
#             warmup_steps=args.warmup_steps,
#             patience=args.patience,
#             mixup_alpha=args.mixup_alpha,
#             augment=use_augment,
#             seed=args.seed,
#             # Log CrossFreqViT knobs too (harmless for other models)
#             fusion_at=args.fusion_at,
#             lf_tokens=args.lf_tokens,
#             hf_bins=args.hf_bins,
#             lf_cutoff=args.lf_cutoff,
#             hf_cutoff=args.hf_cutoff,
#         ),
#     )

#     # Optimizer
#     optimizer = build_optimizer(
#         model,
#         lr=args.lr,
#         weight_decay=args.weight_decay,
#     )

#     # Scheduler (cosine with warmup, stepped per batch)
#     total_steps = args.epochs * len(train_loader)
#     warmup_steps = min(args.warmup_steps, total_steps // 10 if total_steps > 10 else args.warmup_steps)
#     scheduler = build_cosine_scheduler(
#         optimizer,
#         num_warmup_steps=warmup_steps,
#         num_total_steps=total_steps,
#     )

#     # Early stopping
#     os.makedirs(args.checkpoint_dir, exist_ok=True)
#     checkpoint_path = os.path.join(args.checkpoint_dir, f"{run_name}.pth")

#     stopper = EarlyStopping(
#         patience=args.patience,
#         checkpoint_path=checkpoint_path,
#         mode="min",       # monitor val_loss (lower is better)
#         min_delta=0.0,
#     )

#     print(f"[INFO] Starting training for up to {args.epochs} epochs...")
#     print(f"[INFO] Best checkpoint will be saved to: {checkpoint_path}")

#     best_path = None
#     for epoch in range(args.epochs):
#         print(f"\n[Epoch {epoch+1}/{args.epochs}]")

#         val_loss, val_acc = run_epoch_and_log(
#             model=model,
#             train_loader=train_loader,
#             val_loader=val_loader,
#             optimizer=optimizer,
#             scheduler=scheduler,
#             device=device,
#             epoch_idx=epoch,
#             mixup_alpha=args.mixup_alpha,
#         )

#         print(f"[Epoch {epoch+1}] val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")

#         # step early stopping using val_loss
#         should_stop, best_path = stopper.step(val_loss, model)

#         if should_stop:
#             print("[INFO] Early stopping triggered.")
#             break

#     # Safety: if best_path is still None (no improvement?), fall back to last state
#     if best_path is None:
#         best_path = checkpoint_path
#         torch.save(model.state_dict(), best_path)

#     print(f"[INFO] Best model path: {best_path}")

#     # Final test evaluation + wandb logging
#     print("[INFO] Running final evaluation on test set...")
#     _ = evaluate_and_log(
#         model=model,
#         checkpoint_path=best_path,
#         test_loader=test_loader,
#         device=device,
#         class_names=None,   # you can pass class names list later if you want pretty labels
#     )

#     print("[INFO] Done.")
#     wandb.finish()


# if __name__ == "__main__":
#     main()

