# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from timm.layers import Mlp
from timm.models.vision_transformer import Attention, Block
from torch.nn import functional as F

from ..constants import marker_to_gene


# ----------------------------- Marker Embeddings ---------------------------- #
class MarkerEmbeddingGenePT(nn.Module):
    """GenePT-based marker embedding module that utilizes pre-computed GenePT embeddings for markers."""

    def __init__(self, marker_dict, unknown_marker_embed_dim=3072):
        super().__init__()
        self.genept_embeddings = marker_dict
        self.unknown_marker_embeddings = nn.ModuleDict()
        self.unknown_marker_embed_dim = unknown_marker_embed_dim
        self.register_buffer("_device_tracker", torch.empty(0))  # For robust device tracking

        # Initialize embeddings for all markers that don't have GenePT embeddings
        for marker_name in marker_to_gene.keys():
            m_gene = marker_to_gene[marker_name]
            if m_gene not in self.genept_embeddings:
                self.unknown_marker_embeddings[marker_name] = nn.Embedding(1, self.unknown_marker_embed_dim)
                nn.init.xavier_uniform_(self.unknown_marker_embeddings[marker_name].weight)

    def forward(self, marker_names):
        """Generate embeddings for a list of marker names."""
        target_device = self._device_tracker.device

        embeddings = []
        for m in marker_names:
            m_gene = marker_to_gene[m]
            if m_gene in self.genept_embeddings:
                emb_tensor = torch.tensor(self.genept_embeddings[m_gene], device=target_device, dtype=torch.float)
                embeddings.append(emb_tensor)
            else:
                idx_tensor = torch.zeros(1, dtype=torch.long, device=target_device)
                embeddings.append(self.unknown_marker_embeddings[m](idx_tensor).squeeze(0))

        final_embeddings = torch.stack(embeddings)
        return final_embeddings


# ---------------------------------------------------------------------------- #

# --------------------------- Neural network layers -------------------------- #
class MaskedAttention(Attention):
    """Attention mechanism with optional masking."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        fused_attn=True,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.fused_attn = fused_attn

    def forward(self, x, attn_mask=None):
        """Forward pass with optional attention masking."""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if attn_mask is not None:
            attn_mask = attn_mask == 0
            
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, attn_mask=attn_mask
            )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MaskedBlock(Block):
    """Transformer block with masked attention support."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_bias=True,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )
        self.attn = MaskedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

    def forward(self, x, attn_mask=None):
        """Forward pass with optional attention masking."""
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class PatchEmbedChannelFree(nn.Module):
    """Channel agnostic patch embedding module that applies the same 2D convolution to each channel."""

    def __init__(
        self,
        img_size,
        token_size=16,
        embed_dim=256,
        norm_layer=None,
        bias=True,
    ):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.token_size = (token_size, token_size) if isinstance(token_size, int) else token_size
        self.embed_dim = embed_dim

        # Create a single conv layer that will be applied to each channel
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=token_size, stride=token_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Calculate grid size and number of patches
        self.grid_size = (self.img_size[0] // self.token_size[0], self.img_size[1] // self.token_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        """Forward pass of the PatchEmbedChannelFree module."""
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = x.view(B * C, 1, H, W)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x.view(B, C, -1, self.embed_dim)

        x = self.norm(x)
        return x

# ---------------------------------------------------------------------------- #


class MarkerEmbeddingTable(nn.Module):
    """Marker query straight out of a shipped table -- the folded form of MarkerEmbeddingGenePT."""

    def __init__(self, names: list[str], dim: int):
        super().__init__()
        self.names = list(names)
        self._index = {m: i for i, m in enumerate(self.names)}
        self.register_buffer("table", torch.zeros(len(self.names), dim))

    def forward(self, marker_names):
        missing = [m for m in marker_names if m not in self._index]
        if missing:
            raise KeyError(f"no marker table row for {missing}; this model knows {self.names}")
        idx = torch.tensor([self._index[m] for m in marker_names], dtype=torch.long,
                           device=self.table.device)
        return self.table.index_select(0, idx)
