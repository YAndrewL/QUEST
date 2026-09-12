from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .zoo import CKPT_DIR, MODELS, QUEST_EVA_HE_MARKERS

PATCH = 224

class QuestGenerator:
    """A loaded staining model. Call it with H&E patches and a marker panel."""

    def __init__(self, model: str = "quest-semantic", *, ckpt=None, config=None,
                 device: str = "cuda"):
        self.device = device
        if config in ("run-dir", "legacy-export"):
            self.name, self._family = config, "quest"
            self._init_legacy(Path(ckpt), config)
            return
        if model not in MODELS:
            raise KeyError(f"unknown model {model!r}; choose from {sorted(MODELS)}")
        spec = MODELS[model]
        self.name, self._family = model, spec["family"]
        path = Path(ckpt) if ckpt is not None else CKPT_DIR / spec["ckpt"]
        if not path.exists():
            raise SystemExit(f"{path} not found; "
                             f"see Tutorial 0 (tutorials/00_data.ipynb) for where to download it")
        # the H&E channels a masked autoencoder is conditioned on are inputs, not outputs
        self.markers = [m for m in spec["markers"] if m not in QUEST_EVA_HE_MARKERS]
        if self._family == "quest":
            self._init_quest(path, spec)
        else:
            self._init_eva(path, spec)

    # -- construction -------------------------------------------------------------------------
    def _init_quest(self, path, spec):
        from .dynamic_marker_model import QUEST
        from .encoders.uni2 import load_uni2_model

        sd = torch.load(path, map_location="cpu", weights_only=False)
        table = sd.pop("marker_table")
        # the network reads its marker table from a path at construction, so give it one; the rows
        # are what the export folded, and zoo's marker_* flags make them be used verbatim
        tmp = Path(tempfile.gettempdir()) / f"quest_table_{path.stem}.pt"
        if not tmp.exists():
            torch.save({"marker_embeddings":
                        {m: table[i] for i, m in enumerate(spec["markers"])}}, tmp)
        conf = dict(spec["config"])
        conf["marker_embed_path"], conf["marker_embed_dir"] = str(tmp), str(tmp.parent)
        self.model = QUEST(SimpleNamespace(**conf))
        self._load(sd)
        self.model = self.model.eval().to(self.device)
        self.uni2 = load_uni2_model(pretrained=True, device=self.device)

    def _init_eva(self, path, spec):
        from omegaconf import OmegaConf

        from .eva import EvaMAE

        self._names = list(spec["markers"])          # panel rows first, H&E channels last
        conf = OmegaConf.create(dict(spec["config"]))
        conf.ds.marker_names = self._names
        self.model = EvaMAE(conf)
        self._load(torch.load(path, map_location="cpu", weights_only=False))
        self.model = self.model.eval().to(self.device)
        self.token_size, self.img_size = self.model.token_size, self.model.img_size
        self._n_tok = (self.img_size // self.token_size) ** 2

    def _init_legacy(self, path, kind):
        """A training run directory, or a first-generation self-describing export."""
        from omegaconf import OmegaConf

        from .dynamic_marker_model import QUEST
        from .encoders.uni2 import load_uni2_model

        if kind == "legacy-export":
            blob = torch.load(path, map_location="cpu", weights_only=False)
            conf, sd = dict(blob["model_config"]), blob["state_dict"]
            tmp = Path(tempfile.gettempdir()) / f"quest_legacy_{path.stem}.pt"
            if not tmp.exists():
                torch.save(blob["marker_table"], tmp)
            conf["marker_embed_path"], conf["marker_embed_dir"] = str(tmp), str(tmp.parent)
        else:
            conf = OmegaConf.to_container(
                OmegaConf.load(path / "run_config.yaml").model, resolve=True)
            sd = torch.load(path / "best.ckpt", map_location="cpu",
                            weights_only=False)["state_dict"]
            # a legacy run's learned identity table has to be switched on or every marker
            # collapses to the same image, with nothing raising
            if any("marker_identity_embeddings" in k for k in sd):
                conf["marker_identity"] = True
        self.model = QUEST(SimpleNamespace(**conf))
        self._load(sd)
        self.model = self.model.eval().to(self.device)
        self.uni2 = load_uni2_model(pretrained=True, device=self.device)
        self.markers = sorted(self.model.marker_embed.marker_embeddings)

    def _load(self, state_dict: dict):
        """Load weights, stripping the training wrappers' prefixes and their extra buffers."""
        prefix = "model." if self._family == "quest" else "eva."
        sd = {}
        for k, v in state_dict.items():
            if "_lazy_uni2_model" in k or k in ("im_mean", "im_std"):
                continue
            sd[k[len(prefix):] if k.startswith(prefix) else k] = v
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        unexpected = [k for k in unexpected if "_lazy_uni2_model" not in k]
        if unexpected:
            raise ValueError(f"weights do not match {self.name}: unexpected {unexpected[:4]}")
        if [k for k in missing if "marker_embed" in k and "table" in k]:
            raise ValueError(f"{self.name}: no marker table in the checkpoint")

    # -- inference ----------------------------------------------------------------------------
    def known_markers(self) -> set[str]:
        """Marker names this checkpoint can emit. Anything else raises."""
        return set(self.markers)

    @torch.no_grad()
    def __call__(self, he_bhwc: np.ndarray, panel: list[str], batch: int = 8) -> np.ndarray:
        """(B,224,224,3) float [0,1] + marker names -> (B,len(panel),224,224) float [0,1]."""
        he = np.asarray(he_bhwc, np.float32)
        if he.ndim == 3:
            he = he[None]
        if he.shape[1:] != (PATCH, PATCH, 3):
            raise ValueError(f"expected (B,{PATCH},{PATCH},3) H&E, got {he.shape}")
        if he.max() > 1.5:
            raise ValueError("H&E must be float in [0,1]; divide uint8 patches by 255 first")
        panel = list(panel)
        unknown = [m for m in panel if m not in self.known_markers()]
        if unknown:
            raise KeyError(f"{self.name} cannot emit {unknown}; "
                           f"it knows {len(self.markers)} markers")
        step = self._quest_forward if self._family == "quest" else self._eva_forward
        out = [step(he[s:s + batch], panel) for s in range(0, len(he), batch)]
        return np.concatenate(out, 0).astype(np.float32)

    def _quest_forward(self, he, panel):
        from .encoders.uni2 import uni2_patch_tokens
        x = torch.from_numpy(he).permute(0, 3, 1, 2).contiguous().to(self.device)
        tokens, _ = uni2_patch_tokens(self.uni2, x)
        pred = self.model(he_img=x, he_tokens=tokens.float(), marker_out=list(panel),
                          output_size=(PATCH, PATCH), return_aux=False)
        return pred.clamp(0.0, 1.0).float().cpu().numpy()
 
    def _eva_forward(self, he, panel):
        """The masked-autoencoder call, kept behind the same interface."""
        from einops import rearrange

        B = len(he)
        marker_in = list(panel) + list(QUEST_EVA_HE_MARKERS)
        infer_mask = torch.zeros(len(marker_in), self._n_tok, device=self.device)
        infer_mask[: len(panel), :] = 1.0        # reconstruct the marker rows, keep the H&E
        zeros = np.zeros((B, self.img_size, self.img_size, len(panel)), np.float32)
        x = np.concatenate([zeros, 1.0 - he], axis=-1).astype(np.float32)
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(self.device)
        recon, _ = self.model.model.forward(imgs=x, marker_in=[marker_in] * B,
                                            marker_out=[list(panel)] * B,
                                            infer_mask=infer_mask, channel_mask=None)
        mif = rearrange(recon[:, :, 1:, :], "N C (H W) (P1 P2) -> N C (H P1) (W P2)",
                        P1=self.token_size, P2=self.token_size,
                        H=self.img_size // self.token_size, N=B).clamp(0, 1)
        return mif.float().cpu().numpy()
