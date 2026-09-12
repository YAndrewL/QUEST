from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import cohort, cells, celltyping, predict_mif

CACHE_ROOT = Path(__file__).resolve().parent.parent / "cache" / "pathocell_cells"


def cache_dir(model: str = "quest-semantic") -> Path:
    """Per-cell tables are per MODEL -- the virtual arm is that model's prediction, so a shared dire..."""
    return CACHE_ROOT / model
WEIGHTS = Path(__file__).resolve().parent.parent / "checkpoints" / "celltyper_pathocell.npz"


def region_table(name: str, gen=None, panel: list[str] | None = None,
                 classes: str = "coarse15", model: str = "quest-semantic") -> dict:
    """Per-cell means for one region, both arms, scaled per region. Cached on disk."""
    panel = panel or cohort.pathocell_panel(model)
    key = {"coarse15": "types_coarse15", "merged": "types_merged"}[classes]
    CACHE = cache_dir(model)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npz"

    if path.exists():
        z = dict(np.load(path, allow_pickle=True))
        if "types" in z and "types_merged" not in z:      # the original single-vocabulary layout
            z["types_merged"] = z.pop("types")
        if key not in z:
            import h5py
            with h5py.File(cohort.PATHOCELL_HDF / f"{name}.hdf", "r") as f:
                inst = np.asarray(f["gt_inst"][0], np.int32)
                lab = np.asarray(f["gt_ct_coarse" if classes == "coarse15" else "gt_ct"][0])
            ids, codes = _cell_labels(inst, lab)
            if not np.array_equal(ids, z["ids"]):
                raise SystemExit(f"{name}: mask disagrees with the cached cell ids")
            z[key] = np.array([_name(c, classes) or "" for c in codes])
            np.savez_compressed(path, **z)
        if "panel" in z and list(z["panel"]) != list(panel):
            raise SystemExit(f"{path} was built on a different panel "
                             f"({len(z['panel'])} markers, not {len(panel)})")
        return {"real": z["real"], "virt": z["virt"], "ids": z["ids"], "types": z[key]}

    if gen is None:
        raise ValueError(f"{path} not cached and no generator passed")
    he8, inst, ids, real, _ = cohort.load_pathocell_typing(name, classes=classes, panel=panel)
    virt = predict_mif.predict_region(gen, he8, panel)
    Mr, _ = cells.cell_means(real, inst)
    Mv, _ = cells.cell_means(virt, inst)
    Mr = np.stack([cells.scale_p99(Mr[:, i]) for i in range(Mr.shape[1])], 1)
    Mv = np.stack([cells.scale_p99(Mv[:, i]) for i in range(Mv.shape[1])], 1)
    out = {"real": Mr, "virt": Mv, "ids": ids, "panel": np.array(panel)}
    import h5py
    for scheme, k in (("coarse15", "types_coarse15"), ("merged", "types_merged")):
        with h5py.File(cohort.PATHOCELL_HDF / f"{name}.hdf", "r") as f:
            lab = np.asarray(f["gt_ct_coarse" if scheme == "coarse15" else "gt_ct"][0])
        _, codes = _cell_labels(inst, lab)
        out[k] = np.array([_name(c, scheme) or "" for c in codes])
    np.savez_compressed(path, **out)
    return {"real": Mr, "virt": Mv, "ids": ids, "types": out[key]}


def _cell_labels(inst, lab):
    """(ids, majority label code per cell) for one region's instance mask and label map."""
    ids = np.unique(inst)
    ids = ids[ids > 0]
    flat = inst.ravel()
    order = np.argsort(flat, kind="stable")
    lo = np.searchsorted(flat[order], ids, "left")
    hi = np.searchsorted(flat[order], ids, "right")
    lf = lab.ravel()[order]
    return ids, [int(np.bincount(lf[a:b]).argmax()) for a, b in zip(lo, hi)]


def _name(code: int, classes: str):
    if classes == "coarse15":
        return cohort.COARSE_TYPES[code] if code > 0 else None
    return cohort.PATHOCELL_MERGED.get(cohort.PATHOCELL_FINE[code])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit the cell typer on real MIF and save it to checkpoints/.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", default="reg016_B,reg032_B,reg033_B",
                    help="comma-separated regions kept out of the fit -- every region "
                         "any notebook displays has to be listed here, or that "
                         "notebook is showing training data")
    ap.add_argument("--classes", default="coarse15", choices=["coarse15", "merged"],
                    help="label vocabulary; coarse15 is PathoCell's own")
    ap.add_argument("--out", default=None,
                    help="default: checkpoints/celltyper_<model>.npz")
    ap.add_argument("--model", default="quest-semantic",
                    help="which shipped model predicts the virtual arm")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    panel = cohort.pathocell_panel(a.model)
    regions = cohort.pathocell_regions()
    holdout = [h.strip() for h in a.holdout.split(",") if h.strip()]
    bad = [h for h in holdout if h not in regions]
    if bad:
        raise SystemExit(f"not regions of this cohort: {bad}")

    gen = None
    todo = [r for r in regions if not (cache_dir(a.model) / f"{r}.npz").exists()]
    if todo:
        from quest import QuestGenerator
        gen = QuestGenerator(a.model, device=a.device)

    R, V, Y = [], [], []
    for i, name in enumerate(regions):
        t = region_table(name, gen, panel, classes=a.classes, model=a.model)
        print(f"[{i + 1:3d}/{len(regions)}] {name}: {len(t['ids'])} cells", flush=True)
        if name in holdout:
            continue
        R.append(t["real"]); V.append(t["virt"]); Y.append(t["types"])

    R, V = np.concatenate(R), np.concatenate(V)
    Y = np.array([t if t else None for t in np.concatenate(Y)], dtype=object)
    print(f"fitting on {int(sum(v is not None for v in Y)):,} labelled cells "
          f"from {len(regions) - len(holdout)} regions ({', '.join(holdout)} held out)")

    typers = {"real": celltyping.fit_typer(R, Y, panel),
              "virtual": celltyping.fit_typer(V, Y, panel)}
    # the notebook asserts against this, so a refit with a different holdout cannot quietly turn
    # the case figure into a picture of training data
    for t in typers.values():
        t["holdout"] = ",".join(holdout)
        t["scheme"] = a.classes
        t["model"] = a.model
    out = a.out or str(WEIGHTS.parent / f"celltyper_{a.model}.npz")
    celltyping.save_typers(out, **typers)
    print(f"wrote {out}: {list(typers)} x {len(typers['real']['classes'])} classes")


if __name__ == "__main__":
    main()
