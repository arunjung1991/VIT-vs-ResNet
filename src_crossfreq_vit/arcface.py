# src_crossfreq_vit/arcface.py
# Angular-margin ArcFace head for classification with normalized embeddings.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """
    Implements the ArcFace angular margin:
        cos(θ + m) = cosθ * cos m − sinθ * sin m
    Reference:
        Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)

    Args:
        in_features: feature dimension (from backbone)
        out_features: number of classes
        s: scaling factor (default 64.0)
        m: angular margin (default 0.5)
        easy_margin: use simplified margin behavior

    During training:
        forward(x, labels)
    During evaluation:
        forward(x, eval_mode=True)
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # precompute constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor, labels=None, eval_mode: bool = False):
        """
        x: (B, in_features)
        labels: (B,) int tensor of class indices
        eval_mode: if True, skip margin for pure inference
        """
        x = F.normalize(x, dim=1)
        W = F.normalize(self.weight, dim=1)
        cosine = F.linear(x, W)                     # (B, C)
        if eval_mode or labels is None:
            return self.s * cosine

        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits
