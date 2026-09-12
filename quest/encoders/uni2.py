from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def uni2_timm_kwargs() -> dict:
    import timm

    return {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }


def load_uni2_model(pretrained: bool = True, device: str | torch.device = "cpu"):
    import timm

    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h",
        pretrained=pretrained,
        **uni2_timm_kwargs(),
    )
    model.eval()
    return model.to(device)


def preprocess_he_for_uni2(he_nchw: torch.Tensor, target_hw: tuple[int, int] | None = None) -> torch.Tensor:
    """HE in [0,1] RGB -> ImageNet-normalized NCHW."""
    if target_hw is not None and he_nchw.shape[-2:] != target_hw:
        he_nchw = F.interpolate(he_nchw, size=target_hw, mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=he_nchw.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=he_nchw.device).view(1, 3, 1, 1)
    return (he_nchw - mean) / std


@torch.no_grad()
def uni2_patch_tokens(model: nn.Module, he_nchw: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Args: he_nchw: [B, 3, H, W] float in [0,1] (RGB). Returns: tokens [B, N, D], grid_size (Gh, Gw)"""
    inp = preprocess_he_for_uni2(he_nchw.float())
    tokens = model.get_intermediate_layers(inp, n=1, reshape=False)[-1]
    n = tokens.shape[1]
    side = int(n**0.5)
    if side * side != n:
        raise ValueError(f"UNI2 token count {n} is not a square grid.")
    return tokens, (side, side)


class CachedUNI2Tokens(nn.Module):
    """Pass-through adapter for precomputed UNI2 patch tokens."""

    out_dim = 1536

    def __init__(self, grid_size: tuple[int, int] = (16, 16)):
        super().__init__()
        self.grid_size = grid_size

    def forward(self, he_tokens: torch.Tensor) -> dict:
        n = int(he_tokens.shape[1])
        side = int(round(n**0.5))
        if side * side != n:
            raise ValueError(
                f"Cached UNI2 token count {n} is not a square grid; cannot infer grid_size."
            )
        return {"tokens": he_tokens, "grid_size": (side, side), "features": None}
