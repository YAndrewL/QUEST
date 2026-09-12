from __future__ import annotations

import collections
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch


# -- the embedder ---------------------------------------------------------------------------
class EvaEmbedder:
    """Frozen Eva: a (C,H,W) marker stack plus its marker names -> one L2-normalised vector."""

    def __init__(self, device: str = "cuda", ckpt: str | Path | None = None):
        from omegaconf import OmegaConf

        from quest.eva import EvaMAE
        from quest.zoo import CKPT_DIR, QUEST_EVA_CONFIG

        path = Path(ckpt) if ckpt is not None else CKPT_DIR / "Eva_model.ckpt"
        if not path.exists():
            raise SystemExit(
                f"{path} not found. Download the Eva encoder from HuggingFace:\n"
                f"  python -c \"from huggingface_hub import hf_hub_download; import shutil; "
                f"shutil.copy(hf_hub_download('yandrewl/Eva', 'Eva_model.ckpt'), "
                f"'{path}')\"")
        tbl = CKPT_DIR / "eva_marker_table.npz"
        if not tbl.exists():
            raise SystemExit(f"{tbl} not found; it ships with the QUEST checkpoints")

        # The table IS the vocabulary: one row per marker name it was folded for. Reading the
        # names from it rather than from a constant is what removes the GenePT pickle from the
        # picture -- upstream looks a marker up in a 909 MB gene-embedding dictionary at
        # construction, and this file is that lookup already done, for the 129 markers the QUEST
        # models answer to.
        with np.load(tbl, allow_pickle=True) as z:
            self.markers = [str(m) for m in z["markers"]]
            table = torch.from_numpy(z["table"].astype(np.float32))
        if table.shape[0] != len(self.markers):
            raise ValueError(f"{tbl}: {table.shape[0]} rows for {len(self.markers)} markers")

        conf = OmegaConf.create(dict(QUEST_EVA_CONFIG))
        conf.ds.marker_names = self.markers
        self.model = EvaMAE(conf)

        sd = dict(torch.load(path, map_location="cpu", weights_only=False))
        sd["model.marker_embed.table"] = table
        missing, _ = self.model.load_state_dict(sd, strict=False)
        if missing:
            raise ValueError(f"weights missing from the Eva checkpoint: {missing[:4]}")
        self.model = self.model.eval().to(device)
        self.device = device

    def known_markers(self) -> set[str]:
        return set(self.markers)

    @torch.no_grad()
    def __call__(self, stack_chw: np.ndarray, markers: list[str]) -> np.ndarray:
        """(C,H,W) float marker maps + their names -> one L2-normalised embedding."""
        unknown = [m for m in markers if m not in self.known_markers()]
        if unknown:
            raise KeyError(f"Eva does not know {unknown}")
        if len(markers) != len(stack_chw):
            raise ValueError(f"{len(stack_chw)} channels for {len(markers)} markers")
        x = torch.from_numpy(np.asarray(stack_chw, np.float32)[None]).to(self.device)
        out, _ = self.model.model.forward_encoder(x, [list(markers)])
        e = out[:, :, 1:, :].mean(2).squeeze(1).reshape(1, -1)
        return torch.nn.functional.normalize(e, dim=1)[0].float().cpu().numpy()


# -- the cohort -----------------------------------------------------------------------------
def cohort_records(root=None, *, limit: int | None = None) -> tuple[list[dict], list[str]]:
    """Staged patches joined to the cohort's published per-patch cell composition."""
    from . import cohort

    root = Path(root) if root is not None else cohort.STANFORD_PC_ROOT
    rows = [r for r in csv.DictReader(gzip.open(root / "cell_composition.csv.gz", "rt"))
            if r.get("with_staged_patch_data") == "True"]
    types = [c[len("fraction__"):] for c in rows[0] if c.startswith("fraction__")]
    comp = {(r["region_id"], int(r["patch_id"])): r for r in rows}
    man = json.loads((root / "manifest.json").read_text())["patches"]
    out = []
    for p in man:
        r = comp.get((p["region_id"], p["sample_idx"]))
        if r is None:
            continue
        out.append({**p,
                    "cell": np.array([float(r["fraction__" + t]) for t in types], np.float32),
                    "major": r["major_type"]})
    out.sort(key=lambda p: (p["region_id"], p["sample_idx"]))
    return (out[:limit] if limit else out), types


def eva_panel(record, embedder) -> list[str]:
    """This patch's own markers, restricted to Eva's vocabulary. No fixed subset."""
    known = embedder.known_markers()
    return [m for m in record["biomarkers"] if m in known]


# -- the metrics ----------------------------------------------------------------------------
def retrieval_metrics(sim: np.ndarray, comp: np.ndarray, major, region,
                      ks=(1, 2, 3, 5, 10, 20)) -> dict:
    """Score a query x gallery similarity matrix on biology."""
    N = len(sim)
    reg = np.asarray([hash(r) for r in region])
    valid = reg[:, None] != reg[None, :]
    S = np.where(valid, sim, -np.inf)
    order = np.argsort(-S, axis=1)

    C = np.asarray(comp, np.float64)
    Cn = C - C.mean(1, keepdims=True)
    Cn /= np.linalg.norm(Cn, axis=1, keepdims=True) + 1e-8
    P = Cn @ Cn.T                                   # pairwise composition PCC

    maj = np.asarray(major)
    mid = {t: i for i, t in enumerate(sorted(set(maj.tolist())))}
    ml = np.array([mid[x] for x in maj])
    pri = np.array(list(collections.Counter(ml.tolist()).values()), np.float64) / N

    Pm = np.where(valid, P, -np.inf)
    oracle = float(np.mean([P[i, np.argmax(Pm[i])] for i in range(N)]))
    rng = np.random.RandomState(0)
    rand = float(np.mean([P[i, rng.choice(np.where(valid[i])[0])] for i in range(N)]))
    return {"n": int(N), "n_regions": int(len(set(region))),
            "pcc@1": float(np.mean([P[i, order[i, 0]] for i in range(N)])),
            "pcc@5": float(np.mean([P[i, order[i, :5]].mean() for i in range(N)])),
            "oracle_pcc@1": oracle, "random_pcc": rand,
            "major@1": float(np.mean(ml[order[:, 0]] == ml)),
            "major_chance": float((pri ** 2).sum()),
            "p@k": {str(k): float(np.mean(ml[order[:, :k]] == ml[:, None])) for k in ks if k < N}}


# -- display --------------------------------------------------------------------------------
def composite(stack_chw: np.ndarray, markers: list[str], want: list[str],
              ranges: dict | None = None) -> tuple[np.ndarray, dict]:
    """Three markers as one RGB image: `want[0]` -> R, `want[1]` -> G, `want[2]` -> B."""
    out = np.zeros(stack_chw.shape[1:] + (3,), np.float32)
    used = {}
    for i, m in enumerate(want[:3]):
        if m not in markers:
            continue
        a = np.asarray(stack_chw[markers.index(m)], np.float32)
        lo, hi = ranges[m] if ranges and m in ranges else (np.percentile(a, 1),
                                                           np.percentile(a, 99.5))
        out[..., i] = np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)
        used[m] = (float(lo), float(hi))
    return out, used
