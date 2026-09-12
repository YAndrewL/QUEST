from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import PatchEmbed

from quest.encoders.uni2 import CachedUNI2Tokens
from quest.layers import (
    MarkerEmbeddingCompositeExternal,
    MarkerEmbeddingExternalText,
    MarkerEmbeddingGenePT,
    MarkerEmbeddingPlain,
    MaskedBlock,
)
from quest.pos_embed import get_2d_sincos_pos_embed


def _get(conf: Any, key: str, default: Any = None) -> Any:
    return getattr(conf, key, default) if conf is not None else default


_SAFE_INTERPOLATE_MAX_ELEMENTS = 2_000_000_000


def _interpolate_output_size(
    x: torch.Tensor,
    size: int | tuple[int, int] | list[int] | None,
    scale_factor: float | tuple[float, float] | list[float] | None,
) -> tuple[int, int]:
    if size is not None:
        if isinstance(size, int):
            return size, size
        return int(size[-2]), int(size[-1])
    if scale_factor is None:
        return int(x.shape[-2]), int(x.shape[-1])
    if isinstance(scale_factor, (int, float)):
        return math.floor(int(x.shape[-2]) * float(scale_factor)), math.floor(int(x.shape[-1]) * float(scale_factor))
    return math.floor(int(x.shape[-2]) * float(scale_factor[-2])), math.floor(int(x.shape[-1]) * float(scale_factor[-1]))


def _safe_bilinear_interpolate(
    x: torch.Tensor,
    *,
    size: int | tuple[int, int] | list[int] | None = None,
    scale_factor: float | tuple[float, float] | list[float] | None = None,
) -> torch.Tensor:
    if x.is_cuda and x.ndim == 4:
        out_h, out_w = _interpolate_output_size(x, size, scale_factor)
        elements_per_sample = int(x.shape[1]) * int(out_h) * int(out_w)
        if elements_per_sample > 0 and int(x.shape[0]) * elements_per_sample > _SAFE_INTERPOLATE_MAX_ELEMENTS:
            chunk_size = max(1, _SAFE_INTERPOLATE_MAX_ELEMENTS // elements_per_sample)
            chunks = [
                F.interpolate(chunk, size=size, scale_factor=scale_factor, mode="bilinear", align_corners=False)
                for chunk in x.split(chunk_size, dim=0)
            ]
            return torch.cat(chunks, dim=0)
    return F.interpolate(x, size=size, scale_factor=scale_factor, mode="bilinear", align_corners=False)


class HEEncoderBase(nn.Module):
    """Adapter interface for future Virchow / UNI / other pathology FMs."""

    out_dim: int
    grid_size: tuple[int, int]

    def forward(self, he_img: torch.Tensor) -> dict[str, Any]:
        raise NotImplementedError


class SimpleViTHEEncoder(HEEncoderBase):
    """Default ViT patch encoder for RGB H&E."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.out_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.grid_size = self.patch_embed.grid_size
        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.grid_size[0], cls_token=False)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

        self.blocks = nn.ModuleList(
            [
                MaskedBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, he_img: torch.Tensor) -> dict[str, Any]:
        tokens = self.patch_embed(he_img)
        tokens = tokens + self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return {"tokens": tokens, "grid_size": self.grid_size, "features": None}


class ConvUpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample_mode: str = "bilinear"):
        super().__init__()
        self.upsample_mode = upsample_mode.lower()
        if self.upsample_mode == "pixelshuffle":
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
            conv2_in = out_channels
        elif self.upsample_mode in ("bilinear", "interpolate"):
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
            conv2_in = out_channels
        elif self.upsample_mode in ("convtranspose", "transpose", "deconv"):
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
            conv2_in = out_channels
        else:
            raise ValueError(f"Unknown upsample_mode: {upsample_mode}")
        self.refine = nn.Sequential(
            nn.Conv2d(conv2_in, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample_mode in ("bilinear", "interpolate"):
            x = _safe_bilinear_interpolate(x, scale_factor=2)
        return self.refine(self.up(x))


class SingleScalePixelDecoder(nn.Module):
    """Z_he [B,N,D] -> F_he [B,D_pixel,H,W] with learnable conv upsampling."""

    def __init__(
        self,
        he_dim: int,
        pixel_dim: int,
        patch_size: int = 16,
        hidden_dims: tuple[int, ...] = (512, 256, 256),
        upsample_mode: str = "bilinear",
    ):
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")

        if patch_size & (patch_size - 1) == 0:
            num_upsamples = patch_size.bit_length() - 1
        else:
            num_upsamples = max(1, math.ceil(math.log2(patch_size)))
        channels = [he_dim, *hidden_dims, pixel_dim]
        if len(channels) != num_upsamples + 1:
            defaults = list(hidden_dims)
            while len(defaults) < num_upsamples - 1:
                defaults.append(defaults[-1] if defaults else pixel_dim)
            channels = [he_dim, *defaults[: num_upsamples - 1], pixel_dim]

        self.blocks = nn.ModuleList(
            [ConvUpsampleBlock(channels[i], channels[i + 1], upsample_mode=upsample_mode) for i in range(num_upsamples)]
        )
        self.out_norm = nn.BatchNorm2d(pixel_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        grid_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        bsz, num_tokens, dim = tokens.shape
        gh, gw = grid_size
        x = tokens.transpose(1, 2).reshape(bsz, dim, gh, gw)
        for block in self.blocks:
            x = block(x)
        if x.shape[-2:] != output_size:
            x = _safe_bilinear_interpolate(x, size=output_size)
        return self.out_norm(x)


class TimmHEImageSkip(nn.Module):
    """Optional pretrained convolutional H&E skip branch."""

    def __init__(
        self,
        model_name: str,
        pixel_dim: int,
        pretrained: bool = True,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        # An ImageNet-pretrained backbone expects ImageNet-normalized RGB; normal [0,1]
        # H&E is off-distribution without it. Skip normalization only when training the
        # backbone from scratch.
        self.normalize_input = bool(pretrained)
        self.register_buffer("_in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        channels = self.backbone.feature_info.channels()
        self.projections = nn.ModuleList([nn.Conv2d(ch, pixel_dim, 1) for ch in channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(pixel_dim, pixel_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(pixel_dim, pixel_dim, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, he_img: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        he_img = he_img.float()
        if self.normalize_input:
            he_img = (he_img - self._in_mean) / self._in_std
        features = self.backbone(he_img)
        fused = None
        for feature, projection in zip(features, self.projections):
            projected = projection(feature)
            projected = _safe_bilinear_interpolate(projected, size=output_size)
            fused = projected if fused is None else fused + projected
        if fused is None:
            raise RuntimeError("timm H&E skip backbone did not return features.")
        fused = fused / max(1, len(features))
        return self.fuse(fused)


class MarkerToHECrossAttentionLayer(nn.Module):
    """Cross-attention block that keeps marker queries independent from each other."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        query_norm = self.norm1(query)
        attn_out, _ = self.cross_attn(
            query=query_norm,
            key=memory,
            value=memory,
            need_weights=False,
        )
        query = query + self.dropout1(attn_out)

        query_norm = self.norm2(query)
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(query_norm))))
        return query + self.dropout2(ff_out)


class MarkerToHECrossAttentionDecoder(nn.Module):
    """Stacked marker-to-HE decoder without marker-marker self-attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MarkerToHECrossAttentionLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            query = layer(query, memory)
        return query


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HighResHEPixelDecoder(nn.Module):
    """Mask-feature decoder with HE image skips, similar in spirit to Mask2Former pixel decoders."""

    def __init__(
        self,
        he_dim: int,
        pixel_dim: int,
        hidden_dims: tuple[int, ...] = (512, 384, 256),
        upsample_mode: str = "bilinear",
        he_backbone: str = "none",
        he_backbone_pretrained: bool = True,
        he_backbone_trainable: bool = False,
        he_backbone_out_indices: tuple[int, ...] = (0, 1, 2, 3),
    ):
        super().__init__()
        self.use_pretrained_he_backbone = he_backbone.lower() not in ("none", "conv", "simple")
        if self.use_pretrained_he_backbone:
            self.he_backbone = timm.create_model(
                he_backbone,
                pretrained=he_backbone_pretrained,
                features_only=True,
                out_indices=he_backbone_out_indices,
            )
            if not he_backbone_trainable:
                self.he_backbone.eval()
                for param in self.he_backbone.parameters():
                    param.requires_grad = False
            self.he_backbone_trainable = he_backbone_trainable
            channels = self.he_backbone.feature_info.channels()
            self.he_feature_projections = nn.ModuleList([nn.Conv2d(ch, pixel_dim, 1) for ch in channels])
            self.pretrained_token_proj = ConvNormAct(he_dim, pixel_dim)
            self.pretrained_fuse_blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        ConvNormAct(pixel_dim * 2, pixel_dim),
                        ConvNormAct(pixel_dim, pixel_dim),
                    )
                    for _ in channels
                ]
            )
            self.pretrained_out = nn.Sequential(
                ConvNormAct(pixel_dim, pixel_dim),
                nn.Conv2d(pixel_dim, pixel_dim, 1),
            )
            return

        c0 = max(32, pixel_dim // 4)
        c1 = max(64, pixel_dim // 2)
        c2 = pixel_dim
        self.he_stem = nn.Sequential(
            ConvNormAct(3, c0),
            ConvNormAct(c0, c0),
        )
        self.he_down1 = nn.Sequential(
            nn.Conv2d(c0, c1, 3, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(32, c1), num_channels=c1),
            nn.GELU(),
            ConvNormAct(c1, c1),
        )
        self.he_down2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(32, c2), num_channels=c2),
            nn.GELU(),
            ConvNormAct(c2, c2),
        )

        token_hidden = hidden_dims[0] if hidden_dims else pixel_dim
        self.token_proj = ConvNormAct(he_dim, token_hidden)
        self.token_to_56 = nn.Sequential(
            ConvUpsampleBlock(
                token_hidden,
                hidden_dims[1] if len(hidden_dims) > 1 else pixel_dim,
                upsample_mode=upsample_mode,
            ),
            ConvUpsampleBlock(
                hidden_dims[1] if len(hidden_dims) > 1 else pixel_dim,
                pixel_dim,
                upsample_mode=upsample_mode,
            ),
        )
        self.up_56_to_112 = ConvUpsampleBlock(pixel_dim, pixel_dim, upsample_mode=upsample_mode)
        self.up_112_to_224 = ConvUpsampleBlock(pixel_dim, pixel_dim, upsample_mode=upsample_mode)
        self.fuse_56 = nn.Sequential(
            ConvNormAct(pixel_dim + c2, pixel_dim),
            ConvNormAct(pixel_dim, pixel_dim),
        )
        self.fuse_112 = nn.Sequential(
            ConvNormAct(pixel_dim + c1, pixel_dim),
            ConvNormAct(pixel_dim, pixel_dim),
        )
        self.fuse_224 = nn.Sequential(
            ConvNormAct(pixel_dim + c0, pixel_dim),
            ConvNormAct(pixel_dim, pixel_dim),
        )
        self.out = nn.Sequential(
            ConvNormAct(pixel_dim, pixel_dim),
            nn.Conv2d(pixel_dim, pixel_dim, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        grid_size: tuple[int, int],
        output_size: tuple[int, int],
        he_img: torch.Tensor,
    ) -> torch.Tensor:
        bsz, _, dim = tokens.shape
        gh, gw = grid_size
        he_224 = he_img.float()
        if he_224.shape[-2:] != output_size:
            he_224 = _safe_bilinear_interpolate(he_224, size=output_size)
        if self.use_pretrained_he_backbone:
            x = tokens.transpose(1, 2).reshape(bsz, dim, gh, gw)
            x = self.pretrained_token_proj(x)
            # he_img is normal [0,1] H&E; an ImageNet-pretrained backbone expects
            # ImageNet-normalized RGB.
            mean = torch.tensor([0.485, 0.456, 0.406], device=he_224.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=he_224.device).view(1, 3, 1, 1)
            he_backbone_in = (he_224 - mean) / std
            if self.he_backbone_trainable:
                features = self.he_backbone(he_backbone_in)
            else:
                self.he_backbone.eval()
                with torch.no_grad():
                    features = self.he_backbone(he_backbone_in)
            for feature, projection, fuse in zip(
                reversed(features),
                reversed(self.he_feature_projections),
                reversed(self.pretrained_fuse_blocks),
            ):
                projected = projection(feature)
                x = _safe_bilinear_interpolate(x, size=projected.shape[-2:])
                x = fuse(torch.cat([x, projected], dim=1))
            if x.shape[-2:] != output_size:
                x = _safe_bilinear_interpolate(x, size=output_size)
            return self.pretrained_out(x)

        he_0 = self.he_stem(he_224)
        he_1 = self.he_down1(he_0)
        he_2 = self.he_down2(he_1)

        x = tokens.transpose(1, 2).reshape(bsz, dim, gh, gw)
        x = self.token_proj(x)
        x = self.token_to_56(x)
        x = _safe_bilinear_interpolate(x, size=he_2.shape[-2:])
        x = self.fuse_56(torch.cat([x, he_2], dim=1))
        if x.shape[-2] * 2 == he_1.shape[-2] and x.shape[-1] * 2 == he_1.shape[-1]:
            x = self.up_56_to_112(x)
        else:
            x = _safe_bilinear_interpolate(x, size=he_1.shape[-2:])
        x = self.fuse_112(torch.cat([x, he_1], dim=1))
        if x.shape[-2] * 2 == he_0.shape[-2] and x.shape[-1] * 2 == he_0.shape[-1]:
            x = self.up_112_to_224(x)
        else:
            x = _safe_bilinear_interpolate(x, size=he_0.shape[-2:])
        x = self.fuse_224(torch.cat([x, he_0], dim=1))
        return self.out(x)


class QUEST(nn.Module):
    marker_aliases = {"CD3": "CD3e"}

    def __init__(self, conf: Any):
        super().__init__()
        self.conf = conf
        m = _get(conf, "model", conf)

        self.image_size = int(_get(m, "image_size", 224))
        self.patch_size = int(_get(m, "patch_size", 16))
        self.he_encoder_dim = int(_get(m, "he_encoder_dim", 384))
        self.decoder_dim = int(_get(m, "decoder_dim", self.he_encoder_dim))
        self.pixel_dim = int(_get(m, "pixel_dim", 256))
        self.num_cross_attn_layers = int(_get(m, "num_cross_attn_layers", 4))
        self.num_heads = int(_get(m, "num_heads", 8))
        self.use_film = bool(_get(m, "use_film", False))
        self.use_multiscale_pixel_decoder = bool(_get(m, "use_multiscale_pixel_decoder", False))
        self.he_encoder_type = str(_get(m, "he_encoder", "simple_vit")).lower()
        self.keep_marker_residual = bool(_get(m, "keep_marker_residual", True))
        self.use_marker_conv_head = bool(_get(m, "use_marker_conv_head", True))
        self.use_he_image_skip = bool(_get(m, "use_he_image_skip", True))
        self.use_he_pyramid_decoder = bool(_get(m, "use_he_pyramid_decoder", False))
        self.output_activation = str(_get(m, "output_activation", "none")).lower()
        self.mask_queries_per_marker = int(_get(m, "mask_queries_per_marker", 1))
        self.marker_self_attention = bool(_get(m, "marker_self_attention", True))
        self.use_marker_presence_head = bool(_get(m, "use_marker_presence_head", False))
        self.predict_uncertainty = bool(_get(m, "predict_uncertainty", False))
        self.marker_embed_method = str(_get(m, "marker_embed", "plain"))
        self.marker_embed_fusion = str(_get(m, "marker_embed_fusion", "concat")).lower()
        self.use_separate_marker_sources = (
            self.marker_embed_method in ("composite_external", "composite")
            and self.marker_embed_fusion in ("separate_cross_attn", "separate_attention")
        )
        if self.use_separate_marker_sources and self.mask_queries_per_marker > 1:
            raise NotImplementedError("separate marker source attention currently requires mask_queries_per_marker=1.")

        self.he_encoder = self._build_he_encoder(m)
        self.he_encoder_dim = int(getattr(self.he_encoder, "out_dim", self.he_encoder_dim))
        self.he_to_decoder = (
            nn.Linear(self.he_encoder_dim, self.decoder_dim)
            if self.he_encoder_dim != self.decoder_dim
            else nn.Identity()
        )

        cross_attn_mlp_dim = int(_get(m, "cross_attn_mlp_dim", self.decoder_dim * 4))
        dropout = float(_get(m, "dropout", 0.0))
        if self.use_separate_marker_sources:
            self.marker_embed_sources, self.marker_source_dims = self._build_marker_embedding_sources(m)
            self.marker_dim = int(sum(self.marker_source_dims))
            self.marker_embed = None
            self.marker_proj = None
            self.marker_source_projs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(source_dim, self.decoder_dim),
                        nn.LayerNorm(self.decoder_dim),
                        nn.GELU(),
                        nn.Linear(self.decoder_dim, self.decoder_dim),
                        nn.LayerNorm(self.decoder_dim),
                    )
                    for source_dim in self.marker_source_dims
                ]
            )
            self.marker_source_decoders = nn.ModuleList(
                [
                    self._make_marker_decoder(cross_attn_mlp_dim=cross_attn_mlp_dim, dropout=dropout)
                    for _ in self.marker_source_dims
                ]
            )
            self.marker_source_fuse = nn.Sequential(
                nn.Linear(len(self.marker_source_dims) * self.decoder_dim, self.decoder_dim),
                nn.GELU(),
                nn.Linear(self.decoder_dim, self.decoder_dim),
                nn.LayerNorm(self.decoder_dim),
            )
        else:
            self.marker_embed, self.marker_dim = self._build_marker_embedding(m)
            self.marker_proj = self._build_marker_proj(m)
            self.cross_attn_decoder = self._make_marker_decoder(cross_attn_mlp_dim=cross_attn_mlp_dim, dropout=dropout)
        self.query_norm = nn.LayerNorm(self.decoder_dim)
        if self.mask_queries_per_marker > 1:
            self.mask_query_offsets = nn.Parameter(torch.zeros(self.mask_queries_per_marker, self.decoder_dim))
            self.mask_query_weight = nn.Linear(self.decoder_dim, 1)
            nn.init.normal_(self.mask_query_offsets, std=0.02)
        else:
            self.mask_query_offsets = None
            self.mask_query_weight = None

        if self.use_multiscale_pixel_decoder:
            raise NotImplementedError("Multi-scale pixel decoder requires a hierarchical HE encoder adapter.")
        pixel_hidden_dims = tuple(_get(m, "pixel_hidden_dims", (512, 256, 256)))
        pixel_upsample_mode = str(_get(m, "pixel_upsample_mode", "bilinear"))
        if self.use_he_pyramid_decoder:
            self.pixel_decoder = HighResHEPixelDecoder(
                he_dim=self.he_encoder_dim,
                pixel_dim=self.pixel_dim,
                hidden_dims=pixel_hidden_dims,
                upsample_mode=pixel_upsample_mode,
                he_backbone=str(_get(m, "he_pyramid_backbone", "none")),
                he_backbone_pretrained=bool(_get(m, "he_pyramid_backbone_pretrained", True)),
                he_backbone_trainable=bool(_get(m, "he_pyramid_backbone_trainable", False)),
                he_backbone_out_indices=tuple(_get(m, "he_pyramid_backbone_out_indices", (0, 1, 2, 3))),
            )
        else:
            self.pixel_decoder = SingleScalePixelDecoder(
                he_dim=self.he_encoder_dim,
                pixel_dim=self.pixel_dim,
                patch_size=self.patch_size,
                hidden_dims=pixel_hidden_dims,
                upsample_mode=pixel_upsample_mode,
            )
        he_image_skip_backbone = str(_get(m, "he_image_skip_backbone", "none")).lower()
        he_image_skip_pretrained = bool(_get(m, "he_image_skip_pretrained", True))
        if self.use_he_image_skip and he_image_skip_backbone not in ("none", "conv", "simple"):
            self.he_image_skip = TimmHEImageSkip(
                model_name=he_image_skip_backbone,
                pixel_dim=self.pixel_dim,
                pretrained=he_image_skip_pretrained,
            )
        else:
            self.he_image_skip = (
                nn.Sequential(
                    nn.Conv2d(3, self.pixel_dim // 2, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(self.pixel_dim // 2, self.pixel_dim, 3, padding=1),
                    nn.GELU(),
                )
                if self.use_he_image_skip
                else None
            )
        self.marker_kernel = nn.Linear(self.decoder_dim, self.pixel_dim)
        self.marker_logscale_kernel = (
            nn.Linear(self.decoder_dim, self.pixel_dim)
            if self.predict_uncertainty
            else None
        )
        self.film = nn.Linear(self.decoder_dim, self.pixel_dim * 2) if self.use_film else None
        self.marker_presence_head = nn.Linear(self.decoder_dim, 1) if self.use_marker_presence_head else None
        self.marker_conv_head = (
            nn.Sequential(
                nn.Conv2d(self.pixel_dim, self.pixel_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.pixel_dim, 1, 1),
            )
            if self.use_marker_conv_head
            else None
        )
        if self.film is not None:
            self._init_film_layer()

    def _build_marker_proj(self, m: Any) -> nn.Module:
        """Marker-shared projection from the frozen semantic embedding to the query space."""
        hidden = _get(m, "marker_proj_hidden", None)
        if hidden:
            hidden = int(hidden)
            return nn.Sequential(
                nn.Linear(self.marker_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.decoder_dim),
                nn.LayerNorm(self.decoder_dim),
            )
        return nn.Sequential(
            nn.Linear(self.marker_dim, self.decoder_dim),
            nn.LayerNorm(self.decoder_dim),
        )

    def _make_marker_decoder(self, *, cross_attn_mlp_dim: int, dropout: float) -> nn.Module:
        if self.marker_self_attention:
            layer = nn.TransformerDecoderLayer(
                d_model=self.decoder_dim,
                nhead=self.num_heads,
                dim_feedforward=cross_attn_mlp_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerDecoder(layer, num_layers=self.num_cross_attn_layers)
        return MarkerToHECrossAttentionDecoder(
            d_model=self.decoder_dim,
            nhead=self.num_heads,
            dim_feedforward=cross_attn_mlp_dim,
            dropout=dropout,
            num_layers=self.num_cross_attn_layers,
        )

    @staticmethod
    def _init_film_layer_on(linear: nn.Linear, pixel_dim: int) -> None:
        """FiLM: out = gamma * x + beta. Init gamma=1, beta=0."""
        nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)
        with torch.no_grad():
            linear.bias[:pixel_dim].fill_(1.0)

    def _init_film_layer(self) -> None:
        self._init_film_layer_on(self.film, self.pixel_dim)

    def _build_he_encoder(self, m: Any) -> HEEncoderBase:
        if self.he_encoder_type in ("cached_uni2", "uni2_cached"):
            grid = int(self.image_size // 14)
            return CachedUNI2Tokens(grid_size=(grid, grid))
        if self.he_encoder_type in ("simple_vit", "vit"):
            return SimpleViTHEEncoder(
                image_size=self.image_size,
                patch_size=self.patch_size,
                embed_dim=self.he_encoder_dim,
                depth=int(_get(m, "he_encoder_depth", 6)),
                num_heads=int(_get(m, "he_encoder_num_heads", 6)),
                mlp_ratio=float(_get(m, "he_encoder_mlp_ratio", 4.0)),
            )
        raise ValueError(f"Unknown he_encoder: {self.he_encoder_type}")

    def encode_he(
        self,
        he_img: torch.Tensor | None = None,
        he_tokens: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if he_tokens is not None:
            return self.he_encoder(he_tokens)
        if he_img is None:
            raise ValueError("Either he_img or he_tokens must be provided.")
        return self.he_encoder(he_img)

    def _build_marker_embedding(self, m: Any) -> tuple[nn.Module, int]:
        method = _get(m, "marker_embed", "plain")
        marker_dim_value = _get(m, "marker_dim", 256)
        marker_dim = None if marker_dim_value is None else int(marker_dim_value)
        embed_root = Path(_get(m, "marker_embed_dir", "data/marker_embeddings"))
        normalize = bool(_get(m, "marker_normalize", True))
        center = bool(_get(m, "marker_center", False))
        use_marker_identity = bool(_get(m, "marker_identity", False))

        if method == "plain":
            marker_dim = int(marker_dim or 256)
            return MarkerEmbeddingPlain(dim=marker_dim), marker_dim
        if method == "genept":
            marker_dim = int(marker_dim or 3072)
            pkl = embed_root / "GenePT_embedding_v2/GenePT_gene_protein_embedding_model_3_text.pickle"
            with open(pkl, "rb") as f:
                marker_dict = pickle.load(f)
            return (
                MarkerEmbeddingGenePT(
                    marker_dict,
                    marker_dim,
                    normalize=normalize,
                    use_marker_identity=use_marker_identity,
                ),
                marker_dim,
            )
        if method in ("mistral3", "mistral", "external_text"):
            path = Path(_get(m, "marker_embed_path", embed_root / "mistral3_marker_embeddings.pt"))
            return MarkerEmbeddingExternalText.from_file(
                path,
                unknown_marker_embed_dim=marker_dim,
                aliases=self.marker_aliases,
                normalize=normalize,
                center=center,
                use_marker_identity=use_marker_identity,
            )
        if method in ("composite_external", "composite"):
            paths = list(_get(m, "marker_embed_paths", []))
            if not paths:
                raise ValueError("marker_embed=composite_external requires model.marker_embed_paths.")
            return MarkerEmbeddingCompositeExternal.from_files(
                paths,
                unknown_marker_embed_dim=marker_dim,
                aliases=self.marker_aliases,
                normalize=normalize,
                center=center,
                use_marker_identity=use_marker_identity,
            )
        raise ValueError(f"Unknown marker_embed: {method}")

    def _build_marker_embedding_sources(self, m: Any) -> tuple[nn.ModuleList, list[int]]:
        paths = list(_get(m, "marker_embed_paths", []))
        if not paths:
            raise ValueError("separate marker source attention requires model.marker_embed_paths.")
        learnable_by_source = list(_get(m, "marker_embed_learnable_markers_by_source", []) or [])
        normalize = bool(_get(m, "marker_normalize", True))
        center = bool(_get(m, "marker_center", False))
        use_marker_identity = bool(_get(m, "marker_identity", False))
        modules = []
        dims = []
        for source_idx, path in enumerate(paths):
            learnable_markers = learnable_by_source[source_idx] if source_idx < len(learnable_by_source) else None
            module, dim = MarkerEmbeddingExternalText.from_file(
                path,
                unknown_marker_embed_dim=None,
                aliases=self.marker_aliases,
                learnable_markers=learnable_markers,
                normalize=normalize,
                center=center,
                use_marker_identity=use_marker_identity,
            )
            modules.append(module)
            dims.append(int(dim))
        return nn.ModuleList(modules), dims

    def _normalize_markers(self, names: list[str]) -> list[str]:
        return [self.marker_aliases.get(n, n) for n in names]

    def embed_marker_queries(
        self,
        marker_out: list[str] | list[list[str]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not marker_out:
            raise ValueError("marker_out must be non-empty")
        if isinstance(marker_out[0], str):
            names = self._normalize_markers(marker_out)
            emb = self.marker_embed(names).to(device).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            lists = [self._normalize_markers(list(x)) for x in marker_out]
            if len({len(x) for x in lists}) != 1:
                raise ValueError("All samples in a batch must share the same K.")
            emb = torch.stack([self.marker_embed(x).to(device) for x in lists], dim=0)
        return self.marker_proj(emb)

    def _embed_marker_queries_with_source(
        self,
        source: nn.Module,
        marker_out: list[str] | list[list[str]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not marker_out:
            raise ValueError("marker_out must be non-empty")
        if isinstance(marker_out[0], str):
            names = self._normalize_markers(marker_out)
            return source(names).to(device).unsqueeze(0).expand(batch_size, -1, -1)

        lists = [self._normalize_markers(list(x)) for x in marker_out]
        if len({len(x) for x in lists}) != 1:
            raise ValueError("All samples in a batch must share the same K.")
        return torch.stack([source(x).to(device) for x in lists], dim=0)

    def decode_separate_marker_sources(
        self,
        marker_out: list[str] | list[list[str]],
        batch_size: int,
        device: torch.device,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        decoded_sources = []
        for source, proj, decoder in zip(self.marker_embed_sources, self.marker_source_projs, self.marker_source_decoders):
            q_source = self._embed_marker_queries_with_source(source, marker_out, batch_size, device)
            q_source = proj(q_source)
            if self.marker_self_attention:
                q_decoded = decoder(tgt=q_source, memory=memory)
            else:
                q_decoded = decoder(query=q_source, memory=memory)
            if self.keep_marker_residual:
                q_decoded = q_decoded + q_source
            decoded_sources.append(q_decoded)
        return self.query_norm(self.marker_source_fuse(torch.cat(decoded_sources, dim=-1)))

    def forward(
        self,
        he_img: torch.Tensor | None = None,
        marker_out: list[str] | list[list[str]] | None = None,
        he_tokens: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if marker_out is None:
            raise ValueError("marker_out is required")
        if he_tokens is not None:
            bsz = he_tokens.shape[0]
            device = he_tokens.device
            height, width = output_size or (self.image_size, self.image_size)
        else:
            if he_img is None:
                raise ValueError("Either he_img or he_tokens must be provided.")
            bsz, _, height, width = he_img.shape
            device = he_img.device

        he_out = self.encode_he(he_img=he_img, he_tokens=he_tokens)
        z_he = he_out["tokens"]
        grid_size = he_out["grid_size"]

        memory = self.he_to_decoder(z_he)
        if self.use_separate_marker_sources:
            q_updated = self.decode_separate_marker_sources(marker_out, bsz, device, memory)
            num_markers = q_updated.shape[1]
        else:
            q_marker = self.embed_marker_queries(marker_out, bsz, device)
            num_markers = q_marker.shape[1]
            if self.mask_queries_per_marker > 1:
                q_marker = q_marker.unsqueeze(2) + self.mask_query_offsets.to(device).unsqueeze(0).unsqueeze(0)
                q_marker = q_marker.reshape(bsz, num_markers * self.mask_queries_per_marker, self.decoder_dim)
            if self.marker_self_attention:
                q_decoded = self.cross_attn_decoder(tgt=q_marker, memory=memory)
            else:
                q_decoded = self.cross_attn_decoder(query=q_marker, memory=memory)
            if self.keep_marker_residual:
                q_decoded = q_decoded + q_marker
            q_updated = self.query_norm(q_decoded)
        if self.use_he_pyramid_decoder and he_img is not None:
            f_he = self.pixel_decoder(z_he, grid_size, (height, width), he_img)
        else:
            f_he = self.pixel_decoder(z_he, grid_size, (height, width))
        if not self.use_he_pyramid_decoder and self.he_image_skip is not None and he_img is not None:
            he_skip = he_img.float()
            if he_skip.shape[-2:] != (height, width):
                he_skip = _safe_bilinear_interpolate(he_skip, size=(height, width))
            if isinstance(self.he_image_skip, TimmHEImageSkip):
                f_he = f_he + self.he_image_skip(he_skip, (height, width))
            else:
                f_he = f_he + self.he_image_skip(he_skip)

        marker_kernel = self.marker_kernel(q_updated)  # marker-conditioned mean heads
        f_marker = None
        if self.film is not None:
            gamma, beta = self.film(q_updated).chunk(2, dim=-1)
            if self.marker_conv_head is None:
                scaled_kernel = marker_kernel * gamma
                pred = torch.einsum("bkd,bdhw->bkhw", scaled_kernel, f_he)
                pred = pred + (marker_kernel * beta).sum(dim=-1)[..., None, None]
            else:
                f_marker = f_he.unsqueeze(1) * gamma[..., None, None] + beta[..., None, None]
                pred = torch.einsum("bkd,bkdhw->bkhw", marker_kernel, f_marker)
        else:
            pred = torch.einsum("bkd,bdhw->bkhw", marker_kernel, f_he)
            if self.marker_conv_head is not None:
                f_marker = f_he.unsqueeze(1).expand(-1, marker_kernel.shape[1], -1, -1, -1)
        if self.marker_conv_head is not None:
            if f_marker is None:
                raise RuntimeError("marker_conv_head requires marker-conditioned feature maps.")
            bsz, num_queries, channels, height, width = f_marker.shape
            marker_pred = self.marker_conv_head(f_marker.reshape(bsz * num_queries, channels, height, width))
            pred = pred + marker_pred.reshape(bsz, num_queries, height, width)
        log_scale = None
        if self.predict_uncertainty:
            if self.marker_logscale_kernel is None:
                raise RuntimeError("predict_uncertainty=True requires marker_logscale_kernel.")
            # log_scale is morphology-conditioned predictive uncertainty, not an
            # annotated biological ground-truth uncertainty map.
            logscale_kernel = self.marker_logscale_kernel(q_updated)
            log_scale = torch.einsum("bkd,bdhw->bkhw", logscale_kernel, f_he)
            log_scale = log_scale.clamp(min=-8.0, max=6.0)
        if self.mask_queries_per_marker > 1:
            height, width = pred.shape[-2:]
            pred = pred.reshape(bsz, num_markers, self.mask_queries_per_marker, height, width)
            weights = self.mask_query_weight(q_updated).reshape(bsz, num_markers, self.mask_queries_per_marker, 1, 1)
            weights = weights.softmax(dim=2)
            pred = (pred * weights).sum(dim=2)
            if log_scale is not None:
                log_scale = log_scale.reshape(bsz, num_markers, self.mask_queries_per_marker, height, width)
                scale = torch.exp(log_scale)
                scale = (scale * weights).sum(dim=2)
                log_scale = torch.log(scale.clamp_min(1e-6)).clamp(min=-8.0, max=6.0)
        if self.output_activation == "sigmoid":
            pred = pred.sigmoid()
        elif self.output_activation not in ("none", "identity"):
            raise ValueError(f"Unknown output_activation: {self.output_activation}")
        if not return_aux:
            return pred

        aux = {}
        if self.marker_presence_head is not None:
            q_presence = q_updated
            if self.mask_queries_per_marker > 1:
                q_presence = q_presence.reshape(bsz, num_markers, self.mask_queries_per_marker, self.decoder_dim).mean(dim=2)
            aux["marker_presence_logits"] = self.marker_presence_head(q_presence).squeeze(-1)
        if log_scale is not None:
            aux["log_scale"] = log_scale
        return pred, aux
