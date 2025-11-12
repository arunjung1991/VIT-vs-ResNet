# #!/usr/bin/env python3
# import argparse, os, sys, importlib, json, random
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Subset
# from torchvision import datasets, transforms
# from sklearn.manifold import TSNE
# import numpy as np
# import matplotlib.pyplot as plt

# # ---------------- Helper: TSNE compatibility ----------------
# def make_tsne(**kwargs):
#     """Handle sklearn version differences."""
#     try:
#         return TSNE(max_iter=kwargs.pop("n_iter", 1500), **kwargs)
#     except TypeError:
#         max_iter = kwargs.pop("max_iter", 1500)
#         return TSNE(n_iter=max_iter, **kwargs)

# # ---------------- Argument parsing helpers ------------------
# def _smart_split_kv(s: str) -> list[str]:
#     """Split 'k=v,k=v,args={"a":1,"b":2}' by commas ignoring braces/quotes."""
#     parts, buf = [], []
#     depth = 0
#     in_quote = None
#     esc = False
#     for ch in s:
#         if esc:
#             buf.append(ch)
#             esc = False
#             continue
#         if ch == "\\" and in_quote:
#             buf.append(ch)
#             esc = True
#             continue
#         if ch in ("'", '"'):
#             if in_quote is None:
#                 in_quote = ch
#             elif in_quote == ch:
#                 in_quote = None
#             buf.append(ch)
#             continue
#         if ch == "{" and not in_quote:
#             depth += 1
#             buf.append(ch)
#             continue
#         if ch == "}" and not in_quote and depth > 0:
#             depth -= 1
#             buf.append(ch)
#             continue
#         if ch == "," and not in_quote and depth == 0:
#             parts.append("".join(buf).strip())
#             buf = []
#             continue
#         buf.append(ch)
#     if buf:
#         parts.append("".join(buf).strip())
#     return parts

# def parse_model_arg(s: str):
#     """Parse each --models spec safely."""
#     parts = {}
#     for token in _smart_split_kv(s):
#         if "=" in token:
#             k, v = token.split("=", 1)
#             parts[k.strip()] = v.strip()

#     raw_args = parts.get("args")
#     if raw_args is None or raw_args == "":
#         parts["args"] = {}
#     else:
#         if raw_args.startswith("'") and raw_args.endswith("'"):
#             raw_args = raw_args[1:-1]
#         parts["args"] = json.loads(raw_args)
#     return parts

# # ---------------- Model utilities ----------------------------
# def add_path(p):
#     if p and p not in sys.path:
#         sys.path.insert(0, p)

# def load_model(module_path: str, class_name: str, ctor_kwargs):
#     mod = importlib.import_module(module_path)
#     cls = getattr(mod, class_name)
#     return cls(**ctor_kwargs)

# def load_ckpt(model, ckpt_path):
#     if not ckpt_path:
#         return
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
#     cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
#     model.load_state_dict(cleaned, strict=False)

# def get_named_module(model, dotted):
#     m = model
#     for part in dotted.split("."):
#         m = getattr(m, part)
#     return m

# def find_last_linear(model):
#     last = None
#     for m in model.modules():
#         if isinstance(m, nn.Linear):
#             last = m
#     if last is None:
#         raise RuntimeError("No Linear layer found for hook.")
#     return last

# def extract_features(model, loader, device, feature_layer=None):
#     model.eval().to(device)
#     feats, labels = [], []
#     grab = {}

#     target_layer = get_named_module(model, feature_layer) if feature_layer else find_last_linear(model)
#     def hook(m, inp, out):
#         grab["feat"] = inp[0].detach().cpu()
#     handle = target_layer.register_forward_hook(hook)

#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device)
#             _ = model(x)
#             feats.append(grab["feat"])
#             labels.append(y)
#     handle.remove()

#     return torch.cat(feats).numpy(), torch.cat(labels).numpy()

# # ---------------- Data loader --------------------------------
# def build_loader(data_root, split, img_size, per_class, workers, batch_size, n_classes=None, seed=0):
#     random.seed(seed)
#     tfm = transforms.Compose([
#         transforms.Resize(img_size + 32),
#         transforms.CenterCrop(img_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
#     ])
#     ds = datasets.ImageFolder(os.path.join(data_root, split), transform=tfm)

#     all_ids = list(range(len(ds.classes)))
#     if n_classes:
#         random.shuffle(all_ids)
#         sel_ids = sorted(all_ids[:n_classes])
#     else:
#         sel_ids = all_ids

#     by_class = {cid: [] for cid in sel_ids}
#     for idx, (_, y) in enumerate(ds.samples):
#         if y in by_class:
#             by_class[y].append(idx)

#     chosen = []
#     for y, idxs in by_class.items():
#         random.shuffle(idxs)
#         chosen.extend(idxs[:per_class])

#     subset = Subset(ds, sorted(chosen))
#     id_remap = {orig: i for i, orig in enumerate(sel_ids)}
#     compact_names = [ds.classes[o] for o in sel_ids]

#     class RemapWrapper(torch.utils.data.Dataset):
#         def __init__(self, base, id_remap): self.base, self.id_remap = base, id_remap
#         def __len__(self): return len(self.base)
#         def __getitem__(self, i):
#             x, y = self.base[i]
#             return x, id_remap[y]
#     remapped = RemapWrapper(subset, id_remap)
#     loader = DataLoader(remapped, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
#     return loader, compact_names

# # ---------------- t-SNE plotting -----------------------------
# def run_tsne(feats, labels, title, class_names, out):
#     tsne = make_tsne(n_components=2, init="pca", perplexity=35,
#                      learning_rate="auto", n_iter=1500,
#                      metric="cosine", verbose=1)
#     emb = tsne.fit_transform(feats)
#     plt.figure(figsize=(9,7))
#     for c in np.unique(labels):
#         mask = labels == c
#         plt.scatter(emb[mask,0], emb[mask,1], s=10, alpha=0.7, label=class_names[c])
#     plt.legend(markerscale=2, bbox_to_anchor=(1.02,1), loc="upper left")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(out, dpi=250)
#     plt.close()

# # ---------------- Main ---------------------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--repo-roots", type=str, default=".")
#     ap.add_argument("--data-root", type=str, required=True)
#     ap.add_argument("--split", type=str, default="val")
#     ap.add_argument("--n-classes", type=int, default=None)
#     ap.add_argument("--per-class", type=int, default=50)
#     ap.add_argument("--batch-size", type=int, default=64)
#     ap.add_argument("--workers", type=int, default=4)
#     ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#     ap.add_argument("--models", type=str, nargs="+", required=True)
#     ap.add_argument("--out", type=str, default="tsne_joint.png")
#     args = ap.parse_args()

#     for r in args.repo_roots.split(":"):
#         add_path(r)

#     loader, class_names = build_loader(args.data_root, args.split, 224,
#                                        args.per_class, args.workers,
#                                        args.batch_size, args.n_classes, seed=0)

#     all_feats, all_labels, all_tags = [], [], []

#     for spec in args.models:
#         cfg = parse_model_arg(spec)
#         name = cfg["name"]
#         module = cfg["module"]
#         klass = cfg["class"]
#         ckpt = cfg.get("ckpt", "")
#         feature_layer = cfg.get("feature_layer")
#         ctor_args = cfg.get("args", {})

#         print(f"\n=== Loading {name}: {module}.{klass}")
#         model = load_model(module, klass, ctor_args)
#         load_ckpt(model, ckpt)
#         feats, labels = extract_features(model, loader, args.device, feature_layer)

#         run_tsne(feats, labels, f"{name} ({args.split})", class_names, f"tsne_{name}.png")

#         all_feats.append(feats)
#         all_labels.append(labels)
#         all_tags.extend([name]*len(labels))

#     feats = np.vstack(all_feats)
#     labels = np.concatenate(all_labels)
#     tags = np.array(all_tags)

#     tsne = make_tsne(n_components=2, init="pca", perplexity=35,
#                      learning_rate="auto", n_iter=1500,
#                      metric="cosine", verbose=1)
#     emb = tsne.fit_transform(feats)

#     unique_tags = list(dict.fromkeys(tags.tolist()))
#     markers = ["o","s","^","D","P","X"]
#     tag2marker = {t: markers[i%len(markers)] for i,t in enumerate(unique_tags)}

#     plt.figure(figsize=(10,8))
#     for c in np.unique(labels):
#         mask_c = labels == c
#         for t in unique_tags:
#             mt = mask_c & (tags==t)
#             if np.any(mt):
#                 plt.scatter(emb[mt,0], emb[mt,1], s=10, alpha=0.65,
#                             marker=tag2marker[t], label=f"{class_names[c]} • {t}")
#     handles, lbls = plt.gca().get_legend_handles_labels()
#     by_text = dict(zip(lbls, handles))
#     plt.legend(by_text.values(), by_text.keys(), ncol=2, bbox_to_anchor=(1.02,1),
#                loc="upper left", markerscale=2)
#     plt.title(f"Joint t-SNE ({args.split})")
#     plt.tight_layout()
#     plt.savefig(args.out, dpi=250)
#     plt.close()

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import argparse, os, sys, importlib, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------- TSNE version-compat ----------------
def make_tsne(**kwargs):
    """sklearn >=1.6 uses max_iter; older uses n_iter."""
    try:
        return TSNE(max_iter=kwargs.pop("n_iter", 1500), **kwargs)
    except TypeError:
        max_iter = kwargs.pop("max_iter", 1500)
        return TSNE(n_iter=max_iter, **kwargs)

# ---------------- Arg parsing (robust to JSON) --------
def _smart_split_kv(s: str) -> list[str]:
    parts, buf = [], []
    depth = 0
    in_quote = None
    esc = False
    for ch in s:
        if esc:
            buf.append(ch); esc = False; continue
        if ch == "\\" and in_quote:
            buf.append(ch); esc = True; continue
        if ch in ("'", '"'):
            if in_quote is None: in_quote = ch
            elif in_quote == ch: in_quote = None
            buf.append(ch); continue
        if ch == "{" and not in_quote:
            depth += 1; buf.append(ch); continue
        if ch == "}" and not in_quote and depth > 0:
            depth -= 1; buf.append(ch); continue
        if ch == "," and not in_quote and depth == 0:
            parts.append("".join(buf).strip()); buf = []; continue
        buf.append(ch)
    if buf: parts.append("".join(buf).strip())
    return parts

def parse_model_arg(s: str):
    """
    Format:
      name=<tag>, module=<python.module>, class=<ClassName>,
      ckpt=/path/to.ckpt, args='{"kw":"val"}', feature_layer=head.fc
    """
    parts = {}
    for token in _smart_split_kv(s):
        if "=" in token:
            k, v = token.split("=", 1)
            parts[k.strip()] = v.strip()

    raw_args = parts.get("args")
    if raw_args is None or raw_args == "":
        parts["args"] = {}
    else:
        if (raw_args.startswith("'") and raw_args.endswith("'")) or (raw_args.startswith('"') and raw_args.endswith('"')):
            raw_args = raw_args[1:-1]
        parts["args"] = json.loads(raw_args)
    return parts

# ---------------- Model utils -------------------------
def add_path(p):
    if p and p not in sys.path: sys.path.insert(0, p)

def load_model(module_path: str, class_name: str, ctor_kwargs):
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**(ctor_kwargs or {}))

def load_ckpt(model, ckpt_path):
    if not ckpt_path: return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    cleaned = { (k[7:] if k.startswith("module.") else k): v for k,v in state.items() }
    model.load_state_dict(cleaned, strict=False)

def get_named_module(model, dotted):
    m = model
    for part in dotted.split("."):
        m = getattr(m, part)
    return m

def find_last_linear(model):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise RuntimeError("No Linear layer found to hook.")
    return last

def extract_features(model, loader, device, feature_layer=None):
    model.eval().to(device)
    feats, labels = [], []
    grab = {}
    # try requested layer, fall back to last Linear if not found
    try:
        target_layer = get_named_module(model, feature_layer) if feature_layer else find_last_linear(model)
    except Exception:
        target_layer = find_last_linear(model)

    def hook(m, inp, out):
        # penultimate = INPUT to classifier
        grab["feat"] = inp[0].detach().cpu()
    handle = target_layer.register_forward_hook(hook)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            _ = model(x)
            feats.append(grab["feat"])
            labels.append(y)

    handle.remove()
    feats = torch.cat(feats, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return feats, labels

# ---------------- Data loader w/ class remap -----------
def build_loader(data_root, split, img_size, per_class, workers, batch_size, n_classes=None, seed=0):
    random.seed(seed)
    tfm = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(os.path.join(data_root, split), transform=tfm)

    all_ids = list(range(len(ds.classes)))
    if n_classes:
        random.shuffle(all_ids)
        sel_ids = sorted(all_ids[:n_classes])
    else:
        sel_ids = all_ids

    by_class = {cid: [] for cid in sel_ids}
    for idx, (_, y) in enumerate(ds.samples):
        if y in by_class: by_class[y].append(idx)

    chosen = []
    for y, idxs in by_class.items():
        random.shuffle(idxs)
        chosen.extend(idxs[:per_class])

    subset = Subset(ds, sorted(chosen))
    id_remap = {orig: i for i, orig in enumerate(sel_ids)}
    compact_names = [ds.classes[o] for o in sel_ids]

    class RemapWrapper(torch.utils.data.Dataset):
        def __init__(self, base, id_remap): self.base, self.id_remap = base, id_remap
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            x, y = self.base[i]
            return x, id_remap[y]

    remapped = RemapWrapper(subset, id_remap)
    loader = DataLoader(remapped, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    return loader, compact_names

# ---------------- Feature unification for joint plot ---
def unify_feature_dims(feats_list, target_dim=None):
    """
    Project each [Ni x Di] to common K via PCA-per-model, then L2-normalize rows.
    K = target_dim or min(Di).
    """
    Ds = [f.shape[1] for f in feats_list]
    K = target_dim or min(Ds)
    proj_list = []
    for f in feats_list:
        if f.shape[1] == K:
            f_proj = f
        else:
            pca = PCA(n_components=K, svd_solver="auto", random_state=0)
            f_proj = pca.fit_transform(f)
        denom = np.linalg.norm(f_proj, axis=1, keepdims=True) + 1e-9
        f_proj = (f_proj / denom).astype(np.float32)
        proj_list.append(f_proj)
    return proj_list

# ---------------- Plot helpers -------------------------
def safe_perplexity(n_samples, default=35):
    # perplexity must be < n_samples; keep a little margin
    return max(5, min(default, n_samples // 3 if n_samples > 15 else 5))

def run_tsne(feats, labels, title, class_names, out):
    pp = safe_perplexity(feats.shape[0])
    tsne = make_tsne(n_components=2, init="pca", perplexity=pp,
                     learning_rate="auto", n_iter=1500,
                     metric="cosine", verbose=1)
    emb = tsne.fit_transform(feats)
    plt.figure(figsize=(9,7))
    for c in np.unique(labels):
        m = labels == c
        plt.scatter(emb[m,0], emb[m,1], s=10, alpha=0.7, label=class_names[c])
    plt.legend(markerscale=2, bbox_to_anchor=(1.02,1), loc="upper left", borderaxespad=0.)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=250)
    plt.close()

# ---------------- Main ---------------------------------
def main():
    ap = argparse.ArgumentParser(description="t-SNE for multiple models (different repos), with class subsampling.")
    ap.add_argument("--repo-roots", type=str, default=".", help="Colon-separated import roots (e.g., 'src:src_crossfreq_vit').")
    ap.add_argument("--data-root", type=str, required=True, help="ImageFolder root containing split subfolders.")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--n-classes", type=int, default=None, help="Sample this many random classes.")
    ap.add_argument("--per-class", type=int, default=50, help="Images per selected class.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", type=str, nargs="+", required=True, help="One or more model specs (see docstring).")
    ap.add_argument("--out", type=str, default="tsne_joint.png")
    args = ap.parse_args()

    for r in args.repo_roots.split(":"):
        add_path(r)

    loader, class_names = build_loader(args.data_root, args.split, 224,
                                       args.per_class, args.workers, args.batch_size,
                                       n_classes=args.n_classes, seed=0)

    all_feats, all_labels, all_tags = [], [], []

    for spec in args.models:
        cfg = parse_model_arg(spec)
        name = cfg["name"]
        module = cfg["module"]
        klass = cfg["class"]
        ckpt = cfg.get("ckpt", "")
        feature_layer = cfg.get("feature_layer")
        ctor_args = cfg.get("args", {})

        print(f"\n=== Loading {name}: {module}.{klass}")
        model = load_model(module, klass, ctor_args)
        load_ckpt(model, ckpt)

        feats, labels = extract_features(model, loader, args.device, feature_layer)

        # per-model plot (native feature dim)
        run_tsne(feats, labels, f"{name} ({args.split})", class_names, f"tsne_{name}.png")

        all_feats.append(feats)
        all_labels.append(labels)
        all_tags.extend([name] * len(labels))

    # unify feature dims for joint plot
    all_feats = unify_feature_dims(all_feats)  # default target_dim = min(dims)
    feats = np.vstack(all_feats)
    labels = np.concatenate(all_labels)
    tags = np.array(all_tags)

    pp = safe_perplexity(feats.shape[0])
    tsne = make_tsne(n_components=2, init="pca", perplexity=pp,
                     learning_rate="auto", n_iter=1500,
                     metric="cosine", verbose=1)
    emb = tsne.fit_transform(feats)

    unique_tags = list(dict.fromkeys(tags.tolist()))
    markers = ["o","s","^","D","P","X","*","+","v"]
    tag2marker = {t: markers[i % len(markers)] for i, t in enumerate(unique_tags)}

    plt.figure(figsize=(10,8))
    for c in np.unique(labels):
        mask_c = labels == c
        for t in unique_tags:
            mt = mask_c & (tags == t)
            if np.any(mt):
                plt.scatter(emb[mt,0], emb[mt,1], s=10, alpha=0.65,
                            marker=tag2marker[t], label=f"{class_names[c]} • {t}")
    # de-duplicate legend entries
    handles, lbls = plt.gca().get_legend_handles_labels()
    by_text = dict(zip(lbls, handles))
    plt.legend(by_text.values(), by_text.keys(), ncol=2, bbox_to_anchor=(1.02,1),
               loc="upper left", borderaxespad=0., markerscale=2)
    plt.title(f"Joint t-SNE ({args.split})")
    plt.tight_layout()
    plt.savefig(args.out, dpi=250)
    plt.close()

if __name__ == "__main__":
    main()
