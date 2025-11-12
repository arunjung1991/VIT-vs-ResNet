#!/usr/bin/env python3
# Plot train loss & train acc for TWO selected W&B runs on the same figure.
# Saves: combined_curves_<dataset>.png

import argparse
import re
import wandb
import matplotlib.pyplot as plt

def sanitize(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', s)

def get_history(run, keys):
    hist = run.history(keys=keys, pandas=True)
    out = {}
    for k in keys:
        out[k] = hist[k].dropna().tolist() if k in hist else []
    return out

def pick_latest_run(runs, dataset_name_contains, model_name_contains):
    matches = []
    for r in runs:
        cfg = dict(r.config or {})
        ds = str(cfg.get("dataset_name", "")).lower()
        mn = str(cfg.get("model_name", cfg.get("model_id", ""))).lower()
        if dataset_name_contains.lower() in ds and model_name_contains.lower() in mn:
            matches.append(r)
    if not matches:
        return None
    matches.sort(key=lambda r: r.created_at, reverse=True)
    return matches[0]

def fetch_run(api, entity, project, run_name=None, dataset=None, model_match=None):
    if run_name:
        return api.run(f"{entity}/{project}/{run_name}")
    runs = api.runs(f"{entity}/{project}")
    r = pick_latest_run(runs, dataset_name_contains=dataset, model_name_contains=model_match)
    if r is None:
        raise SystemExit(f"No run found for dataset~='{dataset}', model~='{model_match}'")
    return r

def plot_pair(r1, r2, label1, label2, dataset_key, outfile):
    keys = ["_step", "train/loss", "train/acc"]
    h1 = get_history(r1, keys)
    h2 = get_history(r2, keys)

    steps1 = h1.get("_step", list(range(len(h1.get("train/loss", [])))))
    steps2 = h2.get("_step", list(range(len(h2.get("train/loss", [])))))

    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(steps1, h1.get("train/loss", []), label=label1)
    ax1.plot(steps2, h2.get("train/loss", []), label=label2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Train Loss")
    ax1.set_title(f"Train Loss — {dataset_key}")
    ax1.legend()

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(steps1, h1.get("train/acc", []), label=label1)
    ax2.plot(steps2, h2.get("train/acc", []), label=label2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Train Accuracy (%)")
    ax2.set_title(f"Train Accuracy — {dataset_key}")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {outfile}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True, help="W&B entity, e.g., arunjung1991")
    ap.add_argument("--project", required=True, help="W&B project, e.g., vit_vs_resnetv2")
    ap.add_argument("--dataset", required=True, help="dataset key in your configs (e.g., cifar10, imagenet_subset, imagenet_100)")
    # Option A: exact run names
    ap.add_argument("--run1", default=None, help="Exact W&B run name for curve 1 (overrides auto-pick)")
    ap.add_argument("--run2", default=None, help="Exact W&B run name for curve 2 (overrides auto-pick)")
    # Option B: auto-pick most recent by model match (when --run1/--run2 not given)
    ap.add_argument("--model1", default="ResNet-152", help="Match string for run 1 model")
    ap.add_argument("--model2", default="ViT-B/16", help="Match string for run 2 model")
    ap.add_argument("--label1", default=None, help="Legend label for run1")
    ap.add_argument("--label2", default=None, help="Legend label for run2")
    ap.add_argument("--api_key", default=None, help="Optional W&B API key (else env var/login)")
    args = ap.parse_args()

    if args.api_key:
        wandb.login(key=args.api_key)

    api = wandb.Api()

    # Fetch runs: either exact names or auto by dataset+model strings
    r1 = fetch_run(api, args.entity, args.project,
                   run_name=args.run1, dataset=args.dataset, model_match=args.model1)
    r2 = fetch_run(api, args.entity, args.project,
                   run_name=args.run2, dataset=args.dataset, model_match=args.model2)

    # Labels
    label1 = args.label1 or args.model1
    label2 = args.label2 or args.model2

    outfile = f"combined_curves_{sanitize(args.dataset)}.png"
    plot_pair(r1, r2, label1, label2, args.dataset, outfile)

if __name__ == "__main__":
    main()
