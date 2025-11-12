# src_crossfreq_vit/crossfreq_vit.py
# Cross-frequency ViT: inject low- and high-frequency tokens into a ViT-B/16
# backbone from torchvision. Works with ImageNet(-mini) 224×224 pipelines.

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import os  # if not already imported

def _atomic_torch_save(obj, path: str):
    """Write then atomically move to avoid truncated/corrupt cache files."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)



# -----------------------------
# Small frequency utilities
# -----------------------------

def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,3,H,W) or (B,1,H,W) -> (B,1,H,W) luma.
    """
    if x.size(1) == 1:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def _radial_mask(h: int, w: int, cutoff: float, lt: bool = True, device=None):
    """
    Create a radial frequency mask with normalized radius in [0,1].
    cutoff: fraction of Nyquist (0..1)
    lt=True -> low-pass (<= cutoff), lt=False -> high-pass (>= cutoff)
    """
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing="ij",
    )
    rr = torch.sqrt(xx * xx + yy * yy)                # (H, W)
    if lt:
        m = (rr <= cutoff).float()
    else:
        m = (rr >= cutoff).float()
    return torch.fft.fftshift(m)                      # align with FFT layout

# def _radial_mask(h: int, w: int, cutoff: float, *, lowpass: bool, device=None) -> torch.Tensor:
#     """
#     Create a radial mask with normalized radius in [0, 1] and return it in
#     FFT-SHIFTED layout (i.e., DC at center). This pairs with the shifted
#     spectrum used in _ifft_from_mask().

#     cutoff: fraction of Nyquist (0..1)
#     lowpass=True  -> keep r <= cutoff
#     lowpass=False -> keep r >= cutoff  (high-pass)
#     """
#     yy, xx = torch.meshgrid(
#         torch.linspace(-1.0, 1.0, h, device=device),
#         torch.linspace(-1.0, 1.0, w, device=device),
#         indexing="ij",
#     )
#     rr = torch.sqrt(xx * xx + yy * yy)
#     if lowpass:
#         m = (rr <= cutoff).float()
#     else:
#         m = (rr >= cutoff).float()
#     # return SHIFTED mask (centered DC), to multiply with a shifted spectrum
#     return torch.fft.fftshift(m)


def _ifft_from_mask(gray: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a frequency mask (in SHIFTED layout) to a grayscale image and invert.
    gray: (B,1,H,W)
    mask: (H,W) in fftshift layout
    returns real spatial tensor: (B,1,H,W)

    NOTE: We explicitly shift the spectrum, multiply by the shifted mask,
    then unshift before iFFT. This fixes the classic "mask alignment" bug.
    """
    B, _, H, W = gray.shape
    # shift spectrum to center DC
    f = torch.fft.fftshift(torch.fft.fft2(gray.squeeze(1)))   # (B,H,W), shifted
    f_masked = f * mask                                      # broadcast
    # unshift back, then invert
    img = torch.fft.ifft2(torch.fft.ifftshift(f_masked)).real.unsqueeze(1)
    return img


def _radial_histogram(gray: torch.Tensor, bins: int) -> torch.Tensor:
    """
    gray: (B,1,H,W) -> (B, bins) L1-normalized radial spectrum histogram.
    Uses shifted magnitude and a normalized radial grid.
    """
    B, _, H, W = gray.shape
    f = torch.fft.fft2(gray.squeeze(1))                # (B,H,W)
    mag = torch.abs(f)

    # build normalized radius in shifted layout
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=gray.device),
        torch.linspace(-1.0, 1.0, W, device=gray.device),
        indexing="ij",
    )
    rr = torch.sqrt(xx * xx + yy * yy)
    rr = torch.fft.fftshift(rr)
    rr = rr / rr.max().clamp_min(1e-6)                 # [0,1]

    mag_s = torch.fft.fftshift(mag)                    # align with rr
    edges = torch.linspace(0.0, 1.0, bins + 1, device=gray.device)

    hists = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        band = ((rr >= lo) & (rr < hi)).float()        # (H,W)
        area = band.sum().clamp_min(1.0)
        val = (mag_s * band).sum(dim=(1, 2)) / area    # (B,)
        hists.append(val)
    hist = torch.stack(hists, dim=1)                   # (B,bins)
    hist = hist / (hist.sum(dim=1, keepdim=True).clamp_min(1e-6))
    return hist


# -----------------------------
# Cross-frequency ViT wrapper
# -----------------------------

class CrossFreqViT(nn.Module):
    """
    ViT-B/16 backbone (torchvision) with simple frequency-aware token fusion.

    At encoder block `fusion_at`, we create and append:
      • `lf_tokens` low-frequency tokens from an inverse-FFT low-pass image
        (pooled then linearly projected to the ViT embed dim).
      • 1 high-frequency token from a radial spectrum histogram with `hf_bins` bins.

    These tokens receive their own learnable positional embeddings and are
    concatenated to the sequence before the selected encoder block. A learnable
    scalar gate modulates each group’s contribution.

    Final head: standard linear classifier on CLS after all blocks.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        fusion_at: int = 6,          # index in [0..11] for ViT-B/16
        lf_tokens: int = 4,          # number of LF tokens to append
        lf_cutoff: float = 0.15,     # low-pass cutoff (0..1 of Nyquist)
        hf_bins: int = 16,           # radial histogram bins for HF token
        hf_cutoff: float = 0.15,     # reserved (unused in current HF path)
        num_heads: int = 8,          # kept for API compatibility
        drop: float = 0.0,
    ) -> None:
        super().__init__()

        # ---- Base ViT (torchvision) ----
        if pretrained:
            tv_weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        else:
            tv_weights = None
        vit = models.vit_b_16(weights=tv_weights)
        vit.heads.head = nn.Identity()                # we own the classifier
        self.vit = vit
        self.embed_dim = vit.hidden_dim               # 768 for ViT-B/16

        # ---- Config ----
        self.fusion_at = int(fusion_at)
        self.lf_tokens = int(lf_tokens)
        self.lf_cutoff = float(lf_cutoff)
        self.hf_bins = int(hf_bins)
        self.hf_cutoff = float(hf_cutoff)             # currently unused

        # ---- Projections for frequency features -> tokens ----
        # LF: pool low-pass map to (T, T), reduce columns -> (B, T), project to T tokens
        self.lf_linear = nn.Linear(self.lf_tokens, self.lf_tokens * self.embed_dim)

        # HF: single token from histogram
        self.hf_linear = nn.Linear(self.hf_bins, self.embed_dim)

        # Gates (learnable scalars)
        self.lf_gate = nn.Parameter(torch.tensor(0.5))
        self.hf_gate = nn.Parameter(torch.tensor(0.5))

        # Positional embeddings for the extra tokens injected mid-encoder
        extra_tokens = self.lf_tokens + 1  # LF group + 1 HF token
        self.mid_pos_embed = nn.Parameter(torch.zeros(1, extra_tokens, self.embed_dim))
        nn.init.trunc_normal_(self.mid_pos_embed, std=0.02)

        self.mid_drop = nn.Dropout(drop)

        # Final classifier from CLS after all blocks
        self.classifier = nn.Linear(self.embed_dim, num_classes)

        # Convenience
        self.patch_size = 16

    # ---------- Patch embed + pos enc (pre-encoder input) ----------

    def _forward_patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns token sequence with pos embedding & dropout applied,
        i.e., the input to the first encoder block.
    
        Expected input: images (B, 3, H, W)
        """
        B = x.shape[0]
    
        # 1) Patchify images directly (no _process_input; avoid double-processing)
        x_tok = self.vit.conv_proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
    
        # 2) Prepend CLS
        cls_token = self.vit.class_token.expand(B, -1, -1)       # (B, 1, D)
        seq = torch.cat((cls_token, x_tok), dim=1)                # (B, 1+N, D)
    
        # 3) Positional embedding + dropout from backbone
        seq = seq + self.vit.encoder.pos_embedding[:, : seq.size(1), :]
        seq = self.vit.encoder.dropout(seq)
        return seq


    # def _forward_patch_embed(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     x: (B,3,H,W). Returns token sequence with pos embedding & dropout applied,
    #     i.e., the input to the first encoder block: (B, 1+N, D).
    #     """
    #     B = x.size(0)
    #     # ViT helper (handles flattening if needed)
    #     x_proc = self.vit._process_input(x)                           # (B,3,H,W)
    #     # conv projection -> (B, D, H/ps, W/ps) -> (B, N, D)
    #     x_tok = self.vit.conv_proj(x_proc).flatten(2).transpose(1, 2)
    #     # prepend CLS
    #     cls = self.vit.cls_token.expand(B, -1, -1)
    #     x_tok = torch.cat((cls, x_tok), dim=1)                        # (B,1+N,D)
    #     # add positional embedding + dropout
    #     x_tok = x_tok + self.vit.encoder.pos_embedding
    #     x_tok = self.vit.encoder.dropout(x_tok)
    #     return x_tok

    # ---------- Frequency token builders ----------

    # def _build_lf_tokens(self, img: torch.Tensor) -> torch.Tensor:
    #     """
    #     img: (B,3,H,W)
    #     returns: (B, lf_tokens, D)
    #     """
    #     gray = _rgb_to_gray(img)
    #     H, W = gray.shape[-2:]
    #     lp_mask = _radial_mask(H, W, cutoff=self.lf_cutoff, lowpass=True, device=img.device)
    #     lowpass_img = _ifft_from_mask(gray, lp_mask)                  # (B,1,H,W)

    #     # pool to (T,T), then reduce columns -> (B, T)
    #     T = self.lf_tokens
    #     pooled = F.adaptive_avg_pool2d(lowpass_img, (T, T)).squeeze(1)  # (B,T,T)
    #     pooled = pooled.mean(dim=2)                                     # (B,T)

    #     # project to T tokens of dim D
    #     B = pooled.size(0)
    #     tok = self.lf_linear(pooled).view(B, T, self.embed_dim)         # (B,T,D)
    #     tok = torch.sigmoid(self.lf_gate) * tok
    #     return tok

    # def _build_lf_tokens(self, img: torch.Tensor, cache_root: str = "./fft_cache") -> torch.Tensor:
    #     """
    #     Cached low-frequency tokens.
    #     Returns: (B, T, D)
    #     """
    #     import os, hashlib
    #     os.makedirs(os.path.join(cache_root, "lf"), exist_ok=True)

    #     B = img.size(0)
    #     D = self.vit.hidden_dim
    #     T = self.lf_tokens
    #     out_list = []

    #     for b in range(B):
    #         x_b = img[b:b+1]  # (1,3,H,W)
    #         key = hashlib.md5(x_b.detach().cpu().numpy().tobytes()).hexdigest()
    #         cache_path = os.path.join(cache_root, "lf", f"{key}_T{T}_cut{self.lf_cutoff:.3f}.pt")

    #         if os.path.exists(cache_path):
    #             lf_tok_b = torch.load(cache_path, map_location=img.device)
    #             # ---- sanitize shape to (T, D) ----
    #             if lf_tok_b.dim() == 3:
    #                 lf_tok_b = lf_tok_b.squeeze(0)             # (1,T,D) -> (T,D)
    #             if lf_tok_b.dim() == 2 and lf_tok_b.shape[0] == T and lf_tok_b.shape[1] == D:
    #                 pass
    #             else:
    #                 # last resort: reshape if flat
    #                 lf_tok_b = lf_tok_b.view(T, D)
    #         else:
    #             gray = _rgb_to_gray(x_b)                      # (1,1,H,W)
    #             H, W = gray.shape[-2:]
    #             mask = _radial_mask(H, W, cutoff=self.lf_cutoff, lt=True, device=img.device)
    #             lp = _ifft_from_mask(gray, mask)              # (1,1,H,W)

    #             pooled = F.adaptive_avg_pool2d(lp, (T, T)).squeeze(1).mean(dim=2)  # (1,T)
    #             tok = self.lf_linear(pooled)                  # (1, T*D)
    #             lf_tok_b = tok.view(1, T, D).squeeze(0)       # (T,D)
    #             lf_tok_b = torch.sigmoid(self.lf_gate) * lf_tok_b
    #             torch.save(lf_tok_b.detach().cpu(), cache_path)

    #         out_list.append(lf_tok_b.to(img.device, non_blocking=True))  # (T,D)

    #     lf_tok = torch.stack(out_list, dim=0)  # (B,T,D)
    #     return lf_tok
    def _build_lf_tokens(self, img: torch.Tensor, cache_root: str = "./fft_cache") -> torch.Tensor:
        """
        Cached low-frequency tokens (B, lf_tokens, D). Robust to partial/corrupt files.
        """
        import hashlib
        os.makedirs(os.path.join(cache_root, "lf"), exist_ok=True)

        B = img.size(0)
        D = self.vit.hidden_dim
        T = self.lf_tokens
        out_list = []

        for b in range(B):
            x_b = img[b:b+1]  # (1,3,H,W)
            key = hashlib.md5(x_b.detach().cpu().numpy().tobytes()).hexdigest()
            cache_path = os.path.join(cache_root, "lf", f"{key}_T{T}_cut{self.lf_cutoff:.3f}.pt")

            lf_tok_b = None
            if os.path.exists(cache_path):
                try:
                    lf_tok_b = torch.load(cache_path, map_location=img.device)
                    # sanity check
                    if not isinstance(lf_tok_b, torch.Tensor) or lf_tok_b.ndim != 2:
                        lf_tok_b = None
                except Exception:
                    lf_tok_b = None  # force recompute

            if lf_tok_b is None:
                gray = _rgb_to_gray(x_b)  # (1,1,H,W)
                H, W = gray.shape[-2:]
                mask = _radial_mask(H, W, cutoff=self.lf_cutoff, lt=True, device=img.device)  # (H,W)
                lp = _ifft_from_mask(gray, mask)  # (1,1,H,W) real

                pooled = F.adaptive_avg_pool2d(lp, (T, T)).squeeze(1).mean(dim=2)  # (1,T)
                tok = self.lf_linear(pooled)                    # (1, T*D)
                lf_tok_b = tok.view(1, T, D).squeeze(0)         # (T,D)
                lf_tok_b = torch.sigmoid(self.lf_gate) * lf_tok_b
                _atomic_torch_save(lf_tok_b.detach().cpu(), cache_path)

            out_list.append(lf_tok_b.to(img.device, non_blocking=True))

        lf_tok = torch.stack(out_list, dim=0)  # (B,T,D)
        return lf_tok




    # def _build_hf_token(self, img: torch.Tensor) -> torch.Tensor:
    #     """
    #     img: (B,3,H,W)
    #     returns: (B, 1, D)
    #     """
    #     gray = _rgb_to_gray(img)
    #     hist = _radial_histogram(gray, self.hf_bins)                    # (B,bins)
    #     tok = self.hf_linear(hist).unsqueeze(1)                         # (B,1,D)
    #     tok = torch.sigmoid(self.hf_gate) * tok
    #     return tok

    # def _build_hf_token(self, img, cache_dir="./fft_cache/hf"):
    #     """
    #     Compute or load cached high-frequency token.
    #     """
    #     import os, torch
    #     os.makedirs(cache_dir, exist_ok=True)

    #     key = str(torch.sum(img).item())[:10]
    #     cache_path = os.path.join(cache_dir, f"{key}.pt")

    #     if os.path.exists(cache_path):
    #         return torch.load(cache_path, map_location=img.device)

    #     # ----- original FFT code -----
    #     gray = _rgb_to_gray(img)
    #     f = torch.fft.fft2(gray.squeeze(1))
    #     fshift = torch.fft.fftshift(f)
    #     mag = torch.abs(fshift)
    #     cy, cx = mag.shape[-2:]  # not critical exact details
    #     spectrum = mag.view(mag.shape[0], -1).mean(-1, keepdim=True)
    #     hf_tok = self.hf_proj(spectrum.unsqueeze(-1)).transpose(1, 2)
    #     hf_tok = self.hf_gate_act(self.hf_gate) * hf_tok
    # # --------------------------------

    #     torch.save(hf_tok, cache_path)
    #     return hf_tok
    # def _build_hf_token(self, img: torch.Tensor, cache_root: str = "./fft_cache") -> torch.Tensor:
    #     """
    #     Cached high-frequency token from radial spectrum histogram.
    #     Returns: (B, 1, D)
    #     """
    #     import os, hashlib
    #     os.makedirs(os.path.join(cache_root, "hf"), exist_ok=True)

    #     B = img.size(0)
    #     D = self.vit.hidden_dim
    #     out_list = []

    #     for b in range(B):
    #         x_b = img[b:b+1]  # (1,3,H,W)
    #         key = hashlib.md5(x_b.detach().cpu().numpy().tobytes()).hexdigest()
    #         cache_path = os.path.join(cache_root, "hf", f"{key}_bins{self.hf_bins}.pt")

    #         if os.path.exists(cache_path):
    #             hf_tok_b = torch.load(cache_path, map_location=img.device)
    #             # ---- sanitize shape to (D,) ----
    #             if hf_tok_b.dim() >= 2:
    #                 hf_tok_b = hf_tok_b.squeeze()             # remove all singleton dims
    #             hf_tok_b = hf_tok_b.view(-1)                  # (D,)
    #             if hf_tok_b.numel() != D:
    #                 # legacy cache (e.g., (1,D) or (1,1,D)) will be reshaped above;
    #                 # if still mismatched, recompute:
    #                 raise RuntimeError("HF cache shape mismatch; delete cache file.")
    #         else:
    #             gray = _rgb_to_gray(x_b)                      # (1,1,H,W)
    #             hist = _radial_histogram(gray, self.hf_bins)  # (1,bins)
    #             proj = self.hf_linear(hist)                   # (1,D)
    #             proj = torch.sigmoid(self.hf_gate) * proj     # (1,D)
    #             hf_tok_b = proj.squeeze(0).detach().cpu()     # (D,)
    #             torch.save(hf_tok_b, cache_path)

    #         out_list.append(hf_tok_b.to(img.device, non_blocking=True))  # (D,)

    #     hf_tok = torch.stack(out_list, dim=0).unsqueeze(1)  # (B,1,D)
    #     return hf_tok
    def _build_hf_token(self, img: torch.Tensor, cache_root: str = "./fft_cache") -> torch.Tensor:
        """
        Cached high-frequency token (B,1,D). Robust to partial/corrupt files.
        """
        import hashlib
        os.makedirs(os.path.join(cache_root, "hf"), exist_ok=True)
    
        B = img.size(0)
        out_list = []
    
        for b in range(B):
            x_b = img[b:b+1]  # (1,3,H,W)
            key = hashlib.md5(x_b.detach().cpu().numpy().tobytes()).hexdigest()
            cache_path = os.path.join(cache_root, "hf", f"{key}_bins{self.hf_bins}.pt")
    
            hf_tok_b = None
            if os.path.exists(cache_path):
                try:
                    hf_tok_b = torch.load(cache_path, map_location=img.device)  # expected (D,)
                    if not isinstance(hf_tok_b, torch.Tensor) or hf_tok_b.ndim != 1:
                        hf_tok_b = None
                except Exception:
                    hf_tok_b = None  # force recompute
    
            if hf_tok_b is None:
                gray = _rgb_to_gray(x_b)                       # (1,1,H,W)
                hist = _radial_histogram(gray, self.hf_bins)   # (1,bins)
                proj = self.hf_linear(hist).squeeze(0)         # (D,)
                proj = torch.sigmoid(self.hf_gate) * proj
                hf_tok_b = proj
                _atomic_torch_save(hf_tok_b.detach().cpu(), cache_path)
    
            out_list.append(hf_tok_b.to(img.device, non_blocking=True))  # each (D,)
    
        hf_tok = torch.stack(out_list, dim=0).unsqueeze(1)  # (B,1,D)
        return hf_tok
    




    # ---------- Forward ----------

    def _inject_extra_tokens(self, seq: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Build LF/HF tokens, add their positional embeddings + dropout,
        and concatenate to the running sequence.
        """
        lf_tok = self._build_lf_tokens(img)                             # (B,T,D)
        hf_tok = self._build_hf_token(img)                              # (B,1,D)
        extra = torch.cat([lf_tok, hf_tok], dim=1)                      # (B,T+1,D)

        # add mid-level positional embeddings (broadcast batch)
        pos = self.mid_pos_embed[:, : extra.size(1), :]
        extra = self.mid_drop(extra + pos)
        return torch.cat([seq, extra], dim=1)                           # (B,1+N+T+1,D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,224,224) or any H,W divisible by 16 (positional embedding
        in torchvision is learned for 224×224; use 224 for best results).
        returns: logits (B, num_classes)
        """
        seq = self._forward_patch_embed(x)                              # (B,1+N,D)

        # run encoder blocks, injecting at fusion_at
        for i, block in enumerate(self.vit.encoder.layers):
            if i == self.fusion_at:
                seq = self._inject_extra_tokens(seq, x)
            seq = block(seq)

        seq = self.vit.encoder.ln(seq)
        cls = seq[:, 0]
        logits = self.classifier(cls)
        return logits

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return CLS embedding (B, D) after encoder+LN, with the same
        token-injection behavior as in forward(). Useful for ArcFace heads.
        """
        seq = self._forward_patch_embed(x)
        for i, block in enumerate(self.vit.encoder.layers):
            if i == self.fusion_at:
                seq = self._inject_extra_tokens(seq, x)
            seq = block(seq)
        seq = self.vit.encoder.ln(seq)
        return seq[:, 0]                                               # (B,D)
