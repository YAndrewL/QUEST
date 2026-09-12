from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

PANEL8 = ["DAPI", "CD3e", "CD8", "CD20", "CD68", "Ki67", "PDL1", "PanCK"]  # for a fixed case level
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoMarkerEncoder:
    """DINOv3 over one marker map at a time: (B, M, H, W) -> (B, M, 768)."""

    key = "dino_vmif"

    def __init__(self, device: str = "cuda", model: str = "vit_base_patch16_dinov3"):
        import timm
        self.model = timm.create_model(model, pretrained=True, num_classes=0).eval().to(device)
        self.size = int(timm.data.resolve_data_config({}, model=self.model)["input_size"][-1])
        self.device = device

    @torch.no_grad()
    def __call__(self, mif, markers=None, norm: str = "raw") -> np.ndarray:
        """`norm`: `raw` keeps the native [0,1]; `minmax` rescales each map on its own."""
        x = torch.as_tensor(np.asarray(mif, np.float32)).to(self.device)
        if x.ndim == 3:
            x = x[None]
        B, M, H, W = x.shape
        x = x.reshape(B * M, 1, H, W)
        if norm == "minmax":
            f = x.reshape(B * M, -1)
            lo = f.min(1).values.view(-1, 1, 1, 1)
            hi = f.max(1).values.view(-1, 1, 1, 1)
            x = (x - lo) / (hi - lo).clamp_min(1e-6)
        x = x.repeat(1, 3, 1, 1)
        if (H, W) != (self.size, self.size):
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = (x - _IMAGENET_MEAN.to(self.device)) / _IMAGENET_STD.to(self.device)
        return self.model(x).reshape(B, M, -1).float().cpu().numpy()


class EvaPanelEncoder:
    """(B, M, H, W) -> (B, 768), the panel read jointly. Needs `checkpoints/Eva_model.ckpt`."""

    key = "eva_vmif"

    def __init__(self, device: str = "cuda", ckpt=None):
        from .retrieval import EvaEmbedder
        self.emb = EvaEmbedder(device=device, ckpt=ckpt)
        self.device = device

    def known_markers(self) -> set[str]:
        return self.emb.known_markers()

    @torch.no_grad()
    def __call__(self, mif, markers=None, norm: str = "raw") -> np.ndarray:
        markers = list(markers or PANEL8)
        x = np.asarray(mif, np.float32)
        if x.ndim == 3:
            x = x[None]
        if norm == "minmax":
            lo = x.min((2, 3), keepdims=True)
            hi = x.max((2, 3), keepdims=True)
            x = (x - lo) / np.maximum(hi - lo, 1e-6)
        return np.stack([self.emb(p, markers) for p in x])


ENCODERS = {"dino": DinoMarkerEncoder, "eva": EvaPanelEncoder}


def patch_features(gen, he, panel=None, encoder="dino", device: str = "cuda",
                   norm: str = "raw", batch: int = 8) -> np.ndarray:
    """H&E -> virtual MIF -> features. (T, M, 768) for `dino`, (T, 768) for `eva`.

    `encoder`: a key of `ENCODERS`, or an already-built one to reuse its weights.
    """
    panel = list(panel or PANEL8)
    enc = ENCODERS[encoder](device=device) if isinstance(encoder, str) else encoder
    he = np.asarray(he, np.float32)
    if he.ndim == 3:
        he = he[None]
    out = [enc(gen(he[i:i + batch], panel), panel, norm) for i in range(0, len(he), batch)]
    return np.concatenate(out, 0)
