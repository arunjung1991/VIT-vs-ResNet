#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train CrossFreq-ViT on CIFAR-10 or ImageNet(-mini).
Supports optional ArcFace head, cosine LR with warmup, early stopping, and wandb logging.
"""

import os
import argparse
import json
import torch

# from src_crossfreq_vit import (
#     set_seed, get_device, init_wandb,
#     get_dataloaders, get_crossfreq_model,
#     build_optimizer, build_cosine_scheduler,
#     EarlyStopping, run_epoch_and_log,
#     evaluate_full, save_checkpoint,
# )

from src_crossfreq_vit import (
    set_seed, get_device,
    get_dataloaders, get_crossfreq_model,
    build_optimizer, build_cosine_scheduler,
    EarlyStopping, run_epoch_and_log,
    evaluate_full, save_checkpoint,
)

from src_crossfreq_vit.arcface import ArcMarginProduct

# ===============================
# W&B Logging Initialization
# ===============================
def init_wandb(project, model_name, dataset_name, train_frac, config_extra):
    """
    Initialize Weights & Biases (W&B) logging.
    Logs to your personal account (arunjung1991) unless overridden by WANDB_ENTITY env var.
    """
    import wandb, os

    # Use env var if available, else default to your personal account
    entity = os.environ.get("WANDB_ENTITY", "arunjung1991")

    run = wandb.init(
        entity=entity,
        project=project,
        name=f"{model_name}_{dataset_name}_{train_frac:.1f}",
        config=config_extra,
    )

    return run.name



def parse_args():
    p = argparse.ArgumentParser(description="Run CrossFreq-ViT experiments")
    # Data
    p.add_argument("--dataset_name", type=str, default="imagenet_subset",
                   choices=["imagenet_subset", "cifar10"], help="Which dataset to use")
    p.add_argument("--data_root", type=str, required=True, help="Root dir of dataset")
    p.add_argument("--train_frac", type=float, default=1.0, help="Fraction of train split to use (0,1]")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--augment", action="store_true", help="Use train-time augmentation")

    # Model (CrossFreq-ViT)
    p.add_argument("--fusion_at", type=int, default=6, help="Encoder block index to inject extra tokens (0..11)")
    p.add_argument("--lf_tokens", type=int, default=4)
    p.add_argument("--lf_cutoff", type=float, default=0.15)
    p.add_argument("--hf_bins", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--freeze_backbone", action="store_true")

    # ArcFace (optional)
    p.add_argument("--use_arcface", action="store_true")
    p.add_argument("--arc_s", type=float, default=64.0)
    p.add_argument("--arc_m", type=float, default=0.5)
    p.add_argument("--arc_easy_margin", action="store_true")

    # Optimization
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=-1, help="If <0, set to 5% of total steps")
    p.add_argument("--mixup_alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    # Logging / checkpoints
    p.add_argument("--project", type=str, default="crossfreq-vit")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--patience", type=int, default=7, help="Early stopping patience (by val_loss)")
    p.add_argument("--save_best_as", type=str, default="best_crossfreq_vit.pth")
    p.add_argument("--save_final_as", type=str, default="final_crossfreq_vit.pth")
    p.add_argument("--eval_only", action="store_true", help="Skip training; just evaluate a checkpoint with --ckpt")
    p.add_argument("--ckpt", type=str, default=None, help="Optional path to load model weights for eval_only or resume")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = get_device()

    # ---- Data ----
    loaders = get_dataloaders(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        train_frac=args.train_frac,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    test_loader  = loaders["test"]
    num_classes  = loaders["num_classes"]

    # ---- Model ----
    model_kwargs = dict(
        pretrained=True,
        fusion_at=args.fusion_at,
        lf_tokens=args.lf_tokens,
        lf_cutoff=args.lf_cutoff,
        hf_bins=args.hf_bins,
        drop=args.dropout,
    )
    model, model_id = get_crossfreq_model(
        num_classes=num_classes,
        device=device,
        freeze_backbone=args.freeze_backbone,
        model_kwargs=model_kwargs,
    )

    # ---- Optional ArcFace head ----
    arc_head = None
    use_features_fn = False
    if args.use_arcface:
        in_features = model.embed_dim  # CLS dim from CrossFreq-ViT
        arc_head = ArcMarginProduct(
            in_features=in_features,
            out_features=num_classes,
            s=args.arc_s,
            m=args.arc_m,
            easy_margin=args.arc_easy_margin,
        ).to(device)
        use_features_fn = True  # CrossFreq-ViT exposes forward_features()

    # Optionally load checkpoint (eval-only or resume-style)
    if args.ckpt and os.path.isfile(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        try:
            model.load_state_dict(state)
        except Exception:
            # If the checkpoint might be a dict with other keys
            if "model" in state:
                model.load_state_dict(state["model"])
            elif "backbone" in state:  # arcface-style bundle
                model.load_state_dict(state["backbone"])
            else:
                raise

    if args.eval_only:
        model.eval()
        results = evaluate_full(model, test_loader, device)
        cm_path = os.path.join(args.out_dir, "cm_eval_only.png")
        results["confusion_matrix_fig"].savefig(cm_path, dpi=180)
        print(json.dumps({
            "accuracy": results["accuracy"],
            "macro": results["macro_metrics"],
            "confusion_matrix_png": cm_path,
        }, indent=2))
        return

    # ---- Optimizer & Scheduler ----
    if arc_head is None:
        opt = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
        params_for_scheduler = model.parameters()
    else:
        opt = build_optimizer(
            list(model.parameters()) + list(arc_head.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )
        params_for_scheduler = list(model.parameters()) + list(arc_head.parameters())

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.05 * total_steps) if args.warmup_steps < 0 else args.warmup_steps
    sch = build_cosine_scheduler(opt, num_warmup_steps=warmup_steps, num_total_steps=total_steps)

    # ---- Logging ----
    run_name = args.run_name
    if not args.no_wandb:
        run_name = init_wandb(
            project=args.project,
            model_name=model_id,
            dataset_name=args.dataset_name,
            train_frac=args.train_frac,
            config_extra=vars(args),
        )

    # ---- Early Stopping ----
    best_path = os.path.join(args.out_dir, args.save_best_as)
    stopper = EarlyStopping(patience=args.patience, checkpoint_path=best_path, mode="min")

    # ---- Train ----
    for epoch in range(1, args.epochs + 1):
        val_loss, val_acc = run_epoch_and_log(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=opt,
            scheduler=sch,
            device=device,
            epoch_idx=epoch,
            mixup_alpha=args.mixup_alpha,
            arc_head=arc_head,
            use_features_fn=use_features_fn,
        )

        stop, _ = stopper.step(val_loss, model)
        if stop:
            print(f"[EarlyStopping] Stopping at epoch {epoch} (no improvement).")
            break

    # Save final weights
    final_path = os.path.join(args.out_dir, args.save_final_as)
    save_checkpoint(model, final_path)

    # Load best (if any) before test eval
    best_for_eval = best_path if os.path.isfile(best_path) else final_path
    model.load_state_dict(torch.load(best_for_eval, map_location=device))

    # ---- Test eval ----
    results = evaluate_full(model, test_loader, device)
    cm_path = os.path.join(args.out_dir, "cm_test.png")
    results["confusion_matrix_fig"].savefig(cm_path, dpi=180)

    summary = {
        "best_checkpoint": best_for_eval,
        "final_checkpoint": final_path,
        "test_accuracy": results["accuracy"],
        "macro": results["macro_metrics"],
        "confusion_matrix_png": cm_path,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
