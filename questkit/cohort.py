from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_ROOT = Path(os.environ.get("QUEST_DATA")
                 or Path(__file__).resolve().parent.parent / "data")

def _rel(p) -> str:
    """A path as the reader sees it: relative to the repo when it is inside it."""
    p = Path(p)
    try:
        return str(p.relative_to(Path(__file__).resolve().parent.parent))
    except ValueError:
        return str(p)


STANFORD_PC_ROOT = DATA_ROOT / "stanford-pc"    # manifest.json + regions.json + he/ + mif/
BOG_ROOT = DATA_ROOT / "bog-86337"              # viz.npz + cells_cohort, cells_vseg
CRC_METU_ROOT = DATA_ROOT / "crc-metu"          # feats/slide*.npz + labels.csv
PATHOCELL_HDF = DATA_ROOT / "PathoCell"         # one .hdf per region, plus the label maps
                                                # (NOT shipped: 36 GB and public -- fetch_pathocell)


@dataclass
class Patch:
    """One paired H&E / real-MIF patch."""
    region_id: str
    stem: str
    he: np.ndarray            # (224,224,3) float [0,1]
    mif: np.ndarray           # (C,224,224) float [0,1]
    markers: list[str]        # length C, the measured panel of this patch's region

    def marker(self, name: str) -> np.ndarray:
        if name not in self.markers:
            raise KeyError(f"{name} not measured here; this region has {sorted(self.markers)}")
        return self.mif[self.markers.index(name)]


def stanford_pc_index(tissue: str | None = "Pancreas") -> list[dict]:
    """Manifest rows for one tissue of stanford-pc (None = every tissue)."""
    man = json.loads((STANFORD_PC_ROOT / "manifest.json").read_text())
    regions = json.loads((STANFORD_PC_ROOT / "regions.json").read_text())
    rows = man["patches"]
    if tissue is not None:
        rows = [p for p in rows if regions.get(p["region_id"], {}).get("tissue") == tissue]
        if not rows:
            have = sorted({regions.get(p["region_id"], {}).get("tissue")
                           for p in man["patches"]})
            raise SystemExit(f"no patches for tissue={tissue!r}; this cohort has {have}")
    return rows


def load_patch(row: dict) -> Patch:
    """Read one manifest row off disk."""
    he = np.load(STANFORD_PC_ROOT / row["he"])
    mif = np.load(STANFORD_PC_ROOT / row["mif"])
    he = he.astype(np.float32) / 255.0 if he.dtype == np.uint8 else he.astype(np.float32)
    if he.shape[0] == 3:                        # tolerate CHW on disk
        he = np.transpose(he, (1, 2, 0))
    mif = mif.astype(np.float32)
    if mif.shape[-1] == len(row["biomarkers"]):  # tolerate HWC on disk
        mif = np.transpose(mif, (2, 0, 1))
    return Patch(row["region_id"], row["stem"], he, mif, [str(b) for b in row["biomarkers"]])


def tissue_fraction(he: np.ndarray) -> float:
    """Fraction of an H&E patch that is tissue rather than slide background."""
    he = np.asarray(he)
    he = he.astype(np.float32) / 255.0 if he.dtype == np.uint8 else he.astype(np.float32)
    if he.shape[0] == 3:                        # tolerate CHW
        he = np.transpose(he, (1, 2, 0))
    lum = he.mean(-1)
    sat = he.max(-1) - he.min(-1)
    return float(((lum < 0.86) | (sat > 0.10)).mean())


def common_panel(rows: list[dict]) -> list[str]:
    """Markers measured in EVERY listed patch."""
    sets = [set(map(str, r["biomarkers"])) for r in rows]
    return sorted(set.intersection(*sets)) if sets else []


PATHOCELL_REPO = "Kainmueller-Lab/PathoCell"   # HuggingFace dataset; `pathocell_hdf/` inside it
                                               # (FabianReith/phenobench mirrors the same files)
PATHOCELL_AUX = ["CT_mapping.txt", "CT_coarse_mapping.txt", "IHC_channels.txt"]


def fetch_pathocell(regions: list[str] | None = None, dest: Path | None = None) -> Path:
    """Download the PathoCell HDFs from HuggingFace into `data/PathoCell`."""
    from huggingface_hub import snapshot_download

    dest = Path(dest) if dest is not None else PATHOCELL_HDF
    dest.mkdir(parents=True, exist_ok=True)
    want = sorted(regions) if regions is not None else None

    if want is not None:
        missing = [r for r in want if not (dest / f"{r}.hdf").exists()]
        if not missing and all((dest / a).exists() for a in PATHOCELL_AUX):
            print(f"[pathocell] all {len(want)} regions already in {_rel(dest)}")
            return dest
        patterns = [f"pathocell_hdf/{r}.hdf" for r in missing] + \
                   [f"pathocell_hdf/{a}" for a in PATHOCELL_AUX]
        print(f"[pathocell] fetching {len(missing)} region(s) from {PATHOCELL_REPO}")
    else:
        patterns = ["pathocell_hdf/*"]
        print(f"[pathocell] fetching the FULL cohort (~36 GB) from {PATHOCELL_REPO}")

    # the repo keeps these under `pathocell_hdf/`; download into a staging dir and lift that one
    # level, so `data/PathoCell` holds the .hdf files directly and the loaders need no special case
    staging = dest.parent / ".pathocell-download"
    snap = Path(snapshot_download(PATHOCELL_REPO, repo_type="dataset", local_dir=str(staging),
                                  allow_patterns=patterns))
    got = sorted((snap / "pathocell_hdf").glob("*"))
    for f in got:
        f.replace(dest / f.name)
    shutil.rmtree(staging, ignore_errors=True)
    n = len(list(dest.glob("*.hdf")))
    print(f"[pathocell] {_rel(dest)} now holds {n} region(s)")
    return dest


def pathocell_regions() -> list[str]:
    """Region names present locally, in a fixed order. See `fetch_pathocell` to get them."""
    have = sorted(p.stem for p in PATHOCELL_HDF.glob("*.hdf")) if PATHOCELL_HDF.exists() else []
    if not have:
        raise SystemExit(
            f"no PathoCell regions in {PATHOCELL_HDF}.\n"
            f"    from questkit import cohort\n"
            f"    cohort.fetch_pathocell(['reg001_A', 'reg001_B'])   # or regions=None for all")
    return have


COARSE_TYPES = {0: "Background", 1: "B cells", 2: "Macrophages/Monocytes", 3: "Adipocytes",
                4: "Dendritic cells", 5: "T cells", 6: "Granulocytes", 7: "NK cells",
                8: "Nerves", 9: "Plasma cells", 10: "Smooth muscle", 11: "Stroma",
                12: "Tumor cells", 13: "Vasculature/Lymphatics", 14: "Other cells"}


PATHOCELL_ALIAS = {
    "FOXP3": "FoxP3", "Cytokeratin": "PanCK", "CD3": "CD3e", "PD-L1": "PDL1", "PD-1": "PD1",
    "Granzyme B": "GranzymeB", "beta-catenin": "bCatenin", "Na-K-ATPase": "Na/K ATPase",
    "LAG-3": "LAG3", "MUC-1": "MUC1", "IDO-1": "IDO1", "BCL-2": "BCL2", "T-bet": "Tbet",
    "Chromogranin A": "ChromograninA", "HOCHST13": "DAPI", "Collagen IV": "CollagenIV",
}


def pathocell_channels() -> dict:
    """All 58 measured channels: marker name -> 0-based index into `ifl`."""
    out = {}
    for line in (PATHOCELL_HDF / "IHC_channels.txt").read_text().splitlines():
        if ";" not in line:
            continue
        idx, rest = line.split(";", 1)
        nm = rest.split(" - ")[0].strip()
        out[PATHOCELL_ALIAS.get(nm, nm)] = int(idx) - 1
    return out


def pathocell_panel(model: str = "quest-semantic") -> list[str]:
    """Every measured channel the model can emit -- the full panel, not a hand-picked subset."""
    from quest.zoo import MODELS
    known = set(MODELS[model]["markers"])
    return [m for m in pathocell_channels() if m in known]


# the cohort's published 30-class `gt_ct` vocabulary (its own CT_mapping.txt)
PATHOCELL_FINE = {
    0: "background", 1: "B cells", 2: "CD11b+ monocytes", 3: "CD11b+CD68+ macrophages",
    4: "CD11c+ DCs", 5: "CD163+ macrophages", 6: "CD3+ T cells", 7: "CD4+ T cells",
    8: "CD4+ T cells CD45RO+", 9: "CD4+ T cells GATA3+", 10: "CD68+ macrophages",
    11: "CD68+ macrophages GzmB+", 12: "CD68+CD163+ macrophages", 13: "CD8+ T cells",
    14: "NK cells", 15: "Tregs", 16: "adipocytes", 17: "dirt", 18: "granulocytes",
    19: "immune cells", 20: "immune cells / vasculature", 21: "lymphatics", 22: "nerves",
    23: "plasma cells", 24: "smooth muscle", 25: "stroma", 26: "tumor cells",
    27: "tumor cells / immune cells", 28: "undefined", 29: "vasculature"}


PATHOCELL_MERGED = {
    "tumor cells": "Epithelial",
    "CD11b+ monocytes": "Myeloid", "CD11b+CD68+ macrophages": "Myeloid",
    "CD11c+ DCs": "Myeloid", "CD163+ macrophages": "Myeloid", "CD68+ macrophages": "Myeloid",
    "CD68+ macrophages GzmB+": "Myeloid", "CD68+CD163+ macrophages": "Myeloid",
    "granulocytes": "Myeloid",
    "smooth muscle": "Stromal/vascular", "stroma": "Stromal/vascular",
    "vasculature": "Stromal/vascular", "lymphatics": "Stromal/vascular",
    "CD4+ T cells": "CD4 T", "CD4+ T cells CD45RO+": "CD4 T", "CD4+ T cells GATA3+": "CD4 T",
    "Tregs": "CD4 T",
    "CD8+ T cells": "CD8 T", "CD3+ T cells": "CD8 T",
    "B cells": "B lineage", "plasma cells": "B lineage"}


COARSE15 = [COARSE_TYPES[i] for i in range(1, 15)]


def load_pathocell_typing(name: str, classes: str = "coarse15",
                          panel: list[str] | None = None):
    """One region, ready for cell typing: H&E, masks, real MIF and the published labels."""
    import h5py
    from skimage.filters import threshold_otsu

    if classes not in ("coarse15", "merged"):
        raise ValueError(f"classes must be 'coarse15' or 'merged', got {classes!r}")
    chan = pathocell_channels()
    panel = list(panel) if panel is not None else pathocell_panel()
    unknown = [m for m in panel if m not in chan]
    if unknown:
        raise KeyError(f"not measured in this cohort: {unknown}")
    field = "gt_ct_coarse" if classes == "coarse15" else "gt_ct"
    with h5py.File(PATHOCELL_HDF / f"{name}.hdf", "r") as f:
        he = np.transpose(f["img"][:], (1, 2, 0)).astype(np.float32)
        inst = np.asarray(f["gt_inst"][0], np.int32)
        lab = np.asarray(f[field][0])
        real = f["ifl"][[chan[m] for m in panel]].astype(np.float32)
    sd = np.uint16(he.std(axis=-1))
    bg = sd <= threshold_otsu(sd)
    he8 = np.uint8(np.clip(he / np.percentile(he[bg], 99, axis=0), 0.0, 1.0) * 255)

    ids = np.unique(inst)
    ids = ids[ids > 0]
    flat = inst.ravel()
    order = np.argsort(flat, kind="stable")
    lo = np.searchsorted(flat[order], ids, "left")
    hi = np.searchsorted(flat[order], ids, "right")
    lf = lab.ravel()[order]
    code = [int(np.bincount(lf[a:b]).argmax()) for a, b in zip(lo, hi)]
    if classes == "coarse15":
        types = np.array([COARSE_TYPES[c] if c > 0 else None for c in code], dtype=object)
    else:
        types = np.array([PATHOCELL_MERGED.get(PATHOCELL_FINE[c]) for c in code], dtype=object)
    return he8, inst, ids, real, types


def load_crc_metu() -> dict:
    """The fitted 5-fold patient-grouped MIL's out-of-fold output, per patient."""
    return json.loads((CRC_METU_ROOT / "mil_oof_metU.json").read_text())


def load_crc_metu_case(patient_id: str) -> dict:
    """One case study: the H&E patch mosaic and the slide grid it sits on."""
    z = np.load(CRC_METU_ROOT / "cases" / f"{patient_id}.npz", allow_pickle=False)
    return {"he": z["he"], "coords": z["coords"], "stride": int(z["stride"]),
            "patient_id": str(z["patient_id"]), "acquisition_id": str(z["acquisition_id"])}
