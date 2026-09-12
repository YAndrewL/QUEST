from pathlib import Path

import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Block
from torch.nn import functional as F

from quest.constants import marker_to_gene, marker_to_idx


class MarkerEmbeddingPlain(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(len(marker_to_idx), embedding_dim=dim, padding_idx=marker_to_idx["PAD"])
        with torch.no_grad():
            self.embedding.weight[marker_to_idx["PAD"]].zero_()

    def forward(self, marker_names: list[str]) -> torch.Tensor:
        indices = torch.tensor(
            [marker_to_idx[name] for name in marker_names],
            device=self.embedding.weight.device,
        )
        return self.embedding(indices)


class MarkerEmbeddingGenePT(nn.Module):
    """Frozen GenePT gene embeddings, optionally L2-normalized."""

    def __init__(
        self,
        marker_dict: dict,
        unknown_marker_embed_dim: int = 3072,
        normalize: bool = True,
        use_marker_identity: bool = False,
    ):
        super().__init__()
        self.genept_embeddings = marker_dict
        self.unknown_marker_embed_dim = unknown_marker_embed_dim
        self.normalize = bool(normalize)
        self.use_marker_identity = bool(use_marker_identity)
        self.register_buffer("_device_tracker", torch.empty(0))
        if self.use_marker_identity:
            self.marker_identity_embeddings = nn.Embedding(len(marker_to_idx), unknown_marker_embed_dim)
            nn.init.xavier_uniform_(self.marker_identity_embeddings.weight)
        else:
            self.marker_identity_embeddings = None

    def forward(self, marker_names: list[str]) -> torch.Tensor:
        device = self._device_tracker.device
        embeddings = []
        for m in marker_names:
            if m == "PAD":
                embeddings.append(torch.zeros(self.unknown_marker_embed_dim, device=device))
                continue
            m_gene = marker_to_gene[m]
            if m_gene not in self.genept_embeddings:
                raise KeyError(
                    f"No GenePT embedding for marker '{m}' (gene '{m_gene}'). Full marker "
                    "coverage is required; per-ID learnable fallbacks were removed to preserve "
                    "zero-shot capability."
                )
            base = torch.tensor(self.genept_embeddings[m_gene], device=device, dtype=torch.float)
            if self.normalize:
                base = F.normalize(base, dim=-1)
            if self.marker_identity_embeddings is not None:
                marker_idx = torch.tensor(marker_to_idx[m], dtype=torch.long, device=device)
                base = base + self.marker_identity_embeddings(marker_idx)
            embeddings.append(base)
        return torch.stack(embeddings)


class MarkerEmbeddingExternalText(nn.Module):
    """Frozen text/semantic embeddings, optionally centered and L2-normalized."""

    def __init__(
        self,
        marker_embeddings: dict[str, torch.Tensor],
        unknown_marker_embed_dim: int,
        aliases: dict[str, str] | None = None,
        normalize: bool = True,
        center: bool = False,
        use_marker_identity: bool = False,
    ):
        super().__init__()
        self.marker_embeddings = {
            str(name): torch.as_tensor(value, dtype=torch.float).flatten()
            for name, value in marker_embeddings.items()
        }
        self.aliases = aliases or {}
        self.unknown_marker_embed_dim = unknown_marker_embed_dim
        self.normalize = bool(normalize)
        self.center = bool(center)
        self.use_marker_identity = bool(use_marker_identity)
        self.register_buffer("_device_tracker", torch.empty(0))
        if self.center:
            stacked = torch.stack(list(self.marker_embeddings.values()))
            self.register_buffer("embedding_center", stacked.mean(dim=0))
        else:
            self.register_buffer("embedding_center", torch.zeros(unknown_marker_embed_dim))
        if self.use_marker_identity:
            self.marker_identity_embeddings = nn.Embedding(len(marker_to_idx), unknown_marker_embed_dim)
            nn.init.xavier_uniform_(self.marker_identity_embeddings.weight)
        else:
            self.marker_identity_embeddings = None

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        unknown_marker_embed_dim: int | None = None,
        aliases: dict[str, str] | None = None,
        learnable_markers: list[str] | tuple[str, ...] | None = None,
        normalize: bool = True,
        center: bool = False,
        use_marker_identity: bool = False,
    ) -> tuple["MarkerEmbeddingExternalText", int]:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "marker_embeddings" in payload:
            marker_embeddings = payload["marker_embeddings"]
        elif isinstance(payload, dict) and "embeddings" in payload:
            marker_embeddings = payload["embeddings"]
        elif isinstance(payload, dict):
            marker_embeddings = payload
        else:
            raise ValueError(f"Unsupported marker embedding payload in {path}")
        first = next(iter(marker_embeddings.values()))
        dim = int(torch.as_tensor(first).numel())
        # ``learnable_markers`` only makes sense alongside the legacy per-ID identity table;
        # without it, dropping a marker's embedding would violate the full-coverage contract.
        if learnable_markers and use_marker_identity:
            marker_embeddings = dict(marker_embeddings)
            for marker_name in learnable_markers:
                marker_embeddings.pop(str(marker_name), None)
                if aliases:
                    marker_embeddings.pop(str(aliases.get(str(marker_name), str(marker_name))), None)
        embed_dim = int(unknown_marker_embed_dim or dim)
        if embed_dim != dim:
            raise ValueError(f"Configured marker_dim={embed_dim}, but {path} contains dim={dim}")
        return (
            cls(
                marker_embeddings,
                embed_dim,
                aliases=aliases,
                normalize=normalize,
                center=center,
                use_marker_identity=use_marker_identity,
            ),
            embed_dim,
        )

    def _resolve_base(self, marker_name: str) -> torch.Tensor | None:
        if marker_name in self.marker_embeddings:
            return self.marker_embeddings[marker_name]
        resolved = self.aliases.get(marker_name, marker_name)
        if resolved in self.marker_embeddings:
            return self.marker_embeddings[resolved]
        return None

    def forward(self, marker_names: list[str]) -> torch.Tensor:
        device = self._device_tracker.device
        embeddings = []
        for marker_name in marker_names:
            if marker_name == "PAD":
                embeddings.append(torch.zeros(self.unknown_marker_embed_dim, device=device))
                continue
            base = self._resolve_base(marker_name)
            if base is None:
                raise KeyError(
                    f"No semantic embedding for marker '{marker_name}'. Full marker coverage "
                    "is required; per-ID learnable fallbacks were removed to preserve zero-shot "
                    "capability. Add it to the marker embedding file."
                )
            base = base.to(device=device)
            if self.center:
                base = base - self.embedding_center.to(device=device)
            if self.normalize:
                base = F.normalize(base, dim=-1)
            if self.marker_identity_embeddings is not None:
                resolved = self.aliases.get(marker_name, marker_name)
                marker_idx_name = marker_name if marker_name in marker_to_idx else resolved
                marker_idx = torch.tensor(marker_to_idx[marker_idx_name], dtype=torch.long, device=device)
                base = base + self.marker_identity_embeddings(marker_idx)
            embeddings.append(base)
        return torch.stack(embeddings)


class MarkerEmbeddingCompositeExternal(MarkerEmbeddingExternalText):
    """Concatenate several frozen marker embedding sources."""

    @classmethod
    def from_files(
        cls,
        paths: list[str | Path],
        unknown_marker_embed_dim: int | None = None,
        aliases: dict[str, str] | None = None,
        normalize: bool = True,
        center: bool = False,
        use_marker_identity: bool = False,
    ) -> tuple["MarkerEmbeddingCompositeExternal", int]:
        if not paths:
            raise ValueError("Composite marker embedding requires at least one source file.")
        source_embeddings: list[dict[str, torch.Tensor]] = []
        source_dims = []
        for path in paths:
            payload = torch.load(Path(path), map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "marker_embeddings" in payload:
                marker_embeddings = payload["marker_embeddings"]
            elif isinstance(payload, dict) and "embeddings" in payload:
                marker_embeddings = payload["embeddings"]
            elif isinstance(payload, dict):
                marker_embeddings = payload
            else:
                raise ValueError(f"Unsupported marker embedding payload in {path}")
            marker_embeddings = {
                str(name): torch.as_tensor(value, dtype=torch.float).flatten()
                for name, value in marker_embeddings.items()
            }
            first = next(iter(marker_embeddings.values()))
            source_dims.append(int(torch.as_tensor(first).numel()))
            source_embeddings.append(marker_embeddings)

        composite_dim = int(sum(source_dims))
        embed_dim = int(unknown_marker_embed_dim or composite_dim)
        if embed_dim != composite_dim:
            raise ValueError(f"Configured marker_dim={embed_dim}, but composite sources contain dim={composite_dim}")

        all_names = set().union(*(set(source.keys()) for source in source_embeddings))
        composite: dict[str, torch.Tensor] = {}
        for name in all_names:
            parts = []
            for source, dim in zip(source_embeddings, source_dims):
                if name in source:
                    parts.append(source[name])
                else:
                    parts.append(torch.zeros(dim, dtype=torch.float))
            composite[name] = torch.cat(parts, dim=0)
        return (
            cls(
                composite,
                embed_dim,
                aliases=aliases,
                normalize=normalize,
                center=center,
                use_marker_identity=use_marker_identity,
            ),
            embed_dim,
        )


class MaskedAttention(Attention):
    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if attn_mask is not None:
            attn_mask = attn_mask == 0
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, attn_mask=attn_mask
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class MaskedBlock(Block):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, **kwargs):
        norm_layer = kwargs.pop("norm_layer", nn.LayerNorm)
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            **kwargs,
        )
        self.attn = MaskedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            norm_layer=nn.LayerNorm,
        )

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
