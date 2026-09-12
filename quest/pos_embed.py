# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import numpy as np
import torch
import torch.nn as nn


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_di..."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M..."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


# ------------------------- Basic Positional Encoding ------------------------ #
class SinCosPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in the Transformer paper."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """Args: d_model: dimension of the model max_len: maximum sequence length dropout: dropout rate"""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and store as buffer (won't be trained)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encoding to the input tensor."""
        x = x + self.pe[:, x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------- #


# --------------------- Rotary Positional Encoding (Rope) -------------------- #
class RotaryPositionalEmbedding1D(nn.Module):
    """1D Rotary Positional Embedding (RoPE) implementation."""

    def __init__(self, model_dim: int, max_seq_length: int = 1200, temperature: float = 10000.0):
        super(RotaryPositionalEmbedding1D, self).__init__()

        assert model_dim % 2 == 0, "Embedding dimension must be multiple of 2 for 1D positional embedding"
        self.model_dim = model_dim

        possible_positions = torch.arange(max_seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32) * -(torch.log(torch.tensor(temperature)) / model_dim)
        )
        pos = possible_positions * div_term
        sin = torch.sin(pos)
        sin = torch.concat([sin, sin], dim=-1)
        self.register_buffer("sin", sin)
        cos = torch.cos(pos)
        cos = torch.concat([cos, cos], dim=-1)
        self.register_buffer("cos", cos)

    def invert_negate(self, x):
        """Helper function to invert and negate the second half of the input."""
        return torch.cat([-x[..., self.model_dim // 2 :], x[..., : self.model_dim // 2]], dim=-1)

    def forward(self, x, pos):
        """Apply rotary positional encoding to the input tensor."""
        x = x * self.cos[pos] + self.invert_negate(x) * self.sin[pos]
        return x


class RotaryPositionalEmbedding2D(nn.Module):
    """2D Rotary Positional Embedding (RoPE) implementation."""

    def __init__(self, model_dim: int, max_pos: int = 1200, temperature: float = 10000.0):
        super(RotaryPositionalEmbedding2D, self).__init__()

        assert model_dim % 4 == 0, "Embedding dimension must be multiple of 4 for 2D positional embedding"
        self.model_dim = model_dim
        self.rope1d = RotaryPositionalEmbedding1D(model_dim // 2, max_pos, temperature)

    def forward(self, x, pos):
        """Apply 2D rotary positional encoding to the input tensor."""
        d = self.model_dim // 2

        x1 = x[..., :d]
        x2 = x[..., d:]

        x1 = self.rope1d(x1, pos.select(dim=-1, index=0))
        x2 = self.rope1d(x2, pos.select(dim=-1, index=1))

        return torch.cat([x1, x2], dim=-1)


# ---------------------------------------------------------------------------- #
