from __future__ import annotations

import json

import numpy as np

from . import cn_vocab

MPP = 0.3775
K_WIN = 10
BG = "#f7f7f7"
# nine neighbourhood hexes, extended: a vocabulary with more CNs than colours would cycle and give
# two different neighbourhoods the same colour, which is silently wrong on a map
CN_COL = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B", "#B279A2",
          "#9D755D", "#8CD17D", "#B6992D", "#499894", "#D37295", "#86BCB6", "#FABFD2",
          "#79706E", "#6B4C9A", "#2F6B4F"]

FAM_COL = {"epithelial": "#F58518", "proliferating": "#E45756", "T cell": "#4C78A8",
           "myeloid": "#B279A2", "endothelial / lymphatic": "#54A24B",
           "stromal / smooth muscle": "#EECA3B", "leukocyte": "#72B7B2"}
ASSOC_COL = ["#9D755D", "#499894", "#D37295", "#8CD17D", "#B6992D", "#86BCB6", "#FABFD2"]
COMP_COL = {"immune": "#4C78A8", "epithelial": "#F58518", "vascular": "#54A24B",
            "stromal": "#EECA3B", "proliferating": "#E45756", "unresolved": "#D9D4CE"}
UNRES_COL = "#D9D4CE"

RENAME_VIRTUAL = {"cytotoxic T cell": "immune cell"}
RENAME_BOTH = {"endothelial / lymphatic": "endothelial"}


def disp(t: str, virtual: bool = False) -> str:
    """A class name as PRINTED; see `RENAME_*` above."""
    for a, b in RENAME_BOTH.items():
        t = t.replace(a, b)
    if virtual:
        for a, b in RENAME_VIRTUAL.items():
            t = t.replace(a, b)
    return t


# -- features -----------------------------------------------------------------------------------
def scale_per_region(M, reg):
    """Per-region p10/p99 scaling: every region is its own acquisition, so it is its own scale."""
    S = np.empty_like(M)
    for r in np.unique(reg):
        m = np.where(reg == r)[0]
        lo = np.percentile(M[m], 10, axis=0)
        hi = np.percentile(M[m], 99, axis=0)
        S[m] = np.clip((M[m] - lo) / np.maximum(hi - lo, 0.03), 0, 1.5)
    return S


def per_region(fn, M, reg):
    S = np.empty_like(M)
    for r in np.unique(reg):
        m = np.where(reg == r)[0]
        S[m] = fn(M[m])
    return S


def l1(M):
    """Per-cell L1: a cell becomes its marker PROFILE, not its brightness."""
    return M / np.maximum(M.sum(1, keepdims=True), 1e-6)


def windows(reg, xs, ys, tcode, n_types, k=K_WIN):
    """Each cell's window cell-state composition: its k nearest cells, itself included."""
    from sklearn.neighbors import NearestNeighbors
    W = np.zeros((len(reg), n_types), np.float32)
    for r in np.unique(reg):
        m = np.where(reg == r)[0]
        if len(m) < k:
            continue
        _, idx = NearestNeighbors(n_neighbors=k).fit(np.c_[xs[m], ys[m]]).kneighbors(
            np.c_[xs[m], ys[m]])
        oh = np.zeros((len(m), n_types), np.float32)
        ok = tcode[m] >= 0
        oh[np.arange(len(m))[ok], tcode[m][ok]] = 1.0
        W[m] = oh[idx].sum(1) / k
    return W


def zsets(C, markers, zmin=1.0):
    """Each centroid's enriched markers, strongest first, with the z-matrix they came from."""
    Z = (C - C.mean(0)) / np.maximum(C.std(0, ddof=0), 1e-9)
    return [[markers[j] for j in np.argsort(-Z[i]) if Z[i][j] >= zmin]
            for i in range(len(C))], Z


def fit(X, K, seed, n_init=10):
    """Full k-means, clusters renumbered by descending size."""
    from sklearn.cluster import KMeans
    km = KMeans(K, random_state=seed, n_init=n_init).fit(X)
    o = np.argsort(-np.bincount(km.labels_, minlength=K))
    inv = np.empty(K, np.int64)
    inv[o] = np.arange(K)
    return inv[km.labels_.astype(np.int64)], km.cluster_centers_[o]


def annotate(C, markers, zmin=1.0):
    """Name a whole vocabulary from its centroids: (names, family labels, signatures)."""
    sig, Z = zsets(C, markers, zmin)
    names = cn_vocab.name_vocabulary(sig, Z, markers)
    return names, [cn_vocab.family_label(n) for n in names], sig


# -- the two arms -------------------------------------------------------------------------------
def load_arms(root) -> dict:
    """Both arms' per-cell tables. The real arm's cells, and the virtual arm's OWN nuclei."""
    from pathlib import Path
    root = Path(root)
    rids = [r["rid"] for r in json.loads((root / "cn_regions.json").read_text())["esophagus"]]
    with np.load(root / "cells_vseg" / f"{rids[0]}.npz", allow_pickle=True) as z:
        markers = [str(q) for q in z["panel"] if str(q) != "DAPI"]

    def one(sub, key):
        B, R, X, Y = [], [], [], []
        for q, rid in enumerate(rids):
            p = root / sub / f"{rid}.npz"
            if not p.exists():
                raise SystemExit(f"{p} missing -- this arm fits its vocabulary cohort-wide")
            with np.load(p, allow_pickle=True) as z:
                pan = [str(t) for t in z["panel"]]
                B.append(np.asarray(z[key][:, [pan.index(t) for t in markers]], np.float32))
                R.append(np.full(len(z["x"]), q, np.int64))
                X.append(np.asarray(z["x"], np.float64))
                Y.append(np.asarray(z["y"], np.float64))
        return {"M": np.concatenate(B), "reg": np.concatenate(R),
                "x": np.concatenate(X), "y": np.concatenate(Y)}

    out = {"root": root, "rids": rids, "markers": markers,
           "real": one("cells_cohort", "real"), "virt": one("cells_vseg", "virt")}
    with np.load(root / "viz.npz", allow_pickle=True) as v:
        out["he"], out["ds"] = v["he"], int(v["ds"])
    return out


def fit_arm(arms: dict, which: str, K: int, NC: int, *, seed: int = 0,
            basis: str = "states") -> dict:
    """The whole recipe on ONE arm: scale, L1, k-means at K, name, windows, k-means at NC."""
    a, mk = arms[which], arms["markers"]
    S = per_region(l1, scale_per_region(a["M"], a["reg"]), a["reg"])
    t, C = fit(S, K, seed)
    names, fams, sig = annotate(C, mk)
    fam = np.array([fams[q] for q in t])
    if basis == "states":
        code, n_code = t, K
    elif basis == "families":
        uF = sorted(set(fam))
        code, n_code = np.array([uF.index(q) for q in fam]), len(uF)
    else:
        raise ValueError(f"basis must be 'states' or 'families', got {basis!r}")
    c, Wc = fit(windows(a["reg"], a["x"], a["y"], code, n_code), NC, seed)
    return {"arm": which, "K": K, "NC": NC, "basis": basis, "types": t, "names": names,
            "families": fams, "signatures": sig, "fam": fam, "cn": c, "Wc": Wc,
            "centroids": C, "reg": a["reg"], "x": a["x"], "y": a["y"]}


def correspondence(m: dict, v: dict) -> dict:
    """Map the two independently-named vocabularies onto ONE unified set of categories."""
    ka, kb, shared = cn_vocab.correspond_two_tier(set(m["families"]), set(v["families"]))
    key_m = np.array([ka[f][1] if ka[f] in shared else f"{f} (real only)" for f in m["fam"]])
    key_v = np.array([kb[f][1] if kb[f] in shared else f"{f} (virtual only)" for f in v["fam"]])
    unified = sorted(set(key_m) | set(key_v))
    NC = m["NC"]

    Um0 = np.array([[np.mean(key_m[m["cn"] == c] == u) for u in unified] for c in range(NC)])
    Uv0 = np.array([[np.mean(key_v[v["cn"] == c] == u) for u in unified] for c in range(NC)])
    pair = {int(q): int(np.argmin(np.abs(Uv0[q] - Um0).sum(-1))) for q in range(NC)}

    def uni(arm, keys):
        lab_of = {arm["names"][i]: keys[arm["types"] == i][0]
                  for i in range(arm["K"]) if (arm["types"] == i).any()}
        U = np.zeros((NC, len(unified)))
        for c in range(NC):
            w = arm["Wc"][c] / max(arm["Wc"][c].sum(), 1e-9)
            for j, st in enumerate(arm["names"]):
                U[c, unified.index(lab_of[st])] += w[j]
        return U

    # the unified category carries `(real only)` / `(virtual only)` so the CLASS legend can flag a
    # one-sided class, but a neighbourhood annotation built out of it reads
    # `immune-unresolved-endothelial (real only) mixed`, which is nonsense
    short = [q.replace(" (real only)", "").replace(" (virtual only)", "") for q in unified]
    # max_mixed=2 for the unified annotation: a class only ONE arm resolves cannot describe the
    # correspondence the unified categories exist to express
    own_m = cn_vocab.name_cn_vocabulary(m["Wc"], m["names"])
    own_v = cn_vocab.name_cn_vocabulary(v["Wc"], v["names"])
    uni_m = [cn_vocab.name_cn(uni(m, key_m)[c], short, max_mixed=2) for c in range(NC)]
    uni_v = [cn_vocab.name_cn(uni(v, key_v)[c], short, max_mixed=2) for c in range(NC)]
    uni_m = [uni_m[c] if c in set(pair.values()) else own_m[c] for c in range(NC)]

    cats = sorted(set(key_m) | set(key_v))
    level = {ka[f][1] if ka[f] in shared else f"{f} (real only)": ka[f][0]
             for f in set(m["families"])}
    level.update({kb[f][1] if kb[f] in shared else f"{f} (virtual only)": kb[f][0]
                  for f in set(v["families"])})
    assoc = [q for q in cats if q not in FAM_COL and q not in COMP_COL and q != "unresolved"]
    colour = {}
    for q in cats:
        base = q.replace(" (real only)", "").replace(" (virtual only)", "")
        if level.get(q) == "compartment":
            colour[q] = COMP_COL.get(base, "#777777")
        elif base == "unresolved":
            colour[q] = UNRES_COL
        elif base in FAM_COL:
            colour[q] = FAM_COL[base]
        else:
            colour[q] = ASSOC_COL[assoc.index(q) % len(ASSOC_COL)]
    return {"ka": ka, "kb": kb, "shared": shared, "key_m": key_m, "key_v": key_v,
            "unified": unified, "categories": cats, "level": level, "colour": colour,
            "pair": pair, "cn_uni_m": uni_m, "cn_uni_v": uni_v,
            "cn_own_m": own_m, "cn_own_v": own_v,
            "cn_col_m": [CN_COL[c] for c in range(NC)],
            "cn_col_v": [CN_COL[pair[c]] for c in range(NC)]}


def transport(reg_a, xa, ya, reg_b, xb, yb, max_um=5.0):
    """For every cell of A, the index of its nearest cell of B in the same region, or -1."""
    from sklearn.neighbors import NearestNeighbors
    idx = np.full(len(reg_a), -1, np.int64)
    for r in np.unique(reg_a):
        ia = np.where(reg_a == r)[0]
        ib = np.where(reg_b == r)[0]
        if not len(ib):
            continue
        d, j = NearestNeighbors(n_neighbors=1).fit(np.c_[xb[ib], yb[ib]]).kneighbors(
            np.c_[xa[ia], ya[ia]])
        ok = d[:, 0] * MPP <= max_um
        idx[ia[ok]] = ib[j[ok, 0]]
    return idx


def scores(m: dict, v: dict, corr: dict | None = None, *, region=None, max_um: float = 5.0):
    """Agreement between the arms, each number beside the ceiling the transport allows."""
    fwd = transport(m["reg"], m["x"], m["y"], v["reg"], v["x"], v["y"], max_um)
    back = transport(v["reg"], v["x"], v["y"], m["reg"], m["x"], m["y"], max_um)
    safe = np.clip(fwd, 0, None)
    rt = np.where((fwd >= 0) & (back[safe] >= 0), back[safe], -1)
    lab_m = corr["key_m"] if corr is not None else m["fam"]
    lab_v = corr["key_v"] if corr is not None else v["fam"]
    got_f = np.where(fwd >= 0, lab_v[safe], "__none__")
    cn_v = np.array([corr["pair"][int(q)] for q in v["cn"]]) if corr is not None else v["cn"]
    got_c = np.where(fwd >= 0, cn_v[safe], -1)

    sel = np.ones(len(fwd), bool) if region is None else np.asarray(region, bool)
    ok = (rt >= 0) & sel
    frac = ok.sum() / max(sel.sum(), 1)
    matched = (fwd >= 0) & sel
    from sklearn.metrics import adjusted_rand_score as ARI
    return {"n": int(sel.sum()), "n_virtual": int(np.unique(safe[matched]).size),
            "unmatched": float((fwd[sel] < 0).mean()),
            "ceiling_class": float(np.mean(lab_m[rt[ok]] == lab_m[ok]) * frac),
            "ceiling_cn": float(np.mean(m["cn"][rt[ok]] == m["cn"][ok]) * frac),
            "same_class": float(np.mean(lab_m[sel] == got_f[sel])),
            "same_cn": float(np.mean(m["cn"][sel] == got_c[sel])),
            "ari_class": float(ARI(lab_m[matched], got_f[matched])),
            "ari_cn": float(ARI(m["cn"][matched], got_c[matched]))}


# -- the figure ---------------------------------------------------------------------------------
def raster_voronoi(x, y, lab, shape, ds, max_px):
    """Exact Voronoi as a raster: every grid point takes the label of its nearest cell."""
    from scipy.spatial import cKDTree
    H, W = shape[0] // ds, shape[1] // ds
    gy, gx = np.mgrid[0:H, 0:W]
    pts = np.c_[gx.ravel() * ds, gy.ravel() * ds]
    d, j = cKDTree(np.c_[x, y]).query(pts, k=1)
    out = lab[j].astype(np.int16).reshape(H, W)
    out[(d > max_px).reshape(H, W)] = -1
    return out


def fill_gaps(he):
    """Fill display-only mosaic gaps with the glass colour observed at this region's own border."""
    he = np.asarray(he).copy()
    gap = (he == 255).all(-1)
    if not gap.any():
        return he, 0.0
    h, w = he.shape[:2]
    yy, xx = np.indices((h, w))
    bw = max(12, min(h, w) // 12)
    border = (~gap) & ((xx < bw) | (xx >= w - bw) | (yy < bw) | (yy >= h - bw))
    sample = he[border]
    if len(sample) > 200:
        bright = sample.mean(-1)
        tone = sample[bright >= np.percentile(bright, 90)].mean(0)
    else:
        tone = np.array([250.0, 250.0, 250.0])
    he[gap] = np.clip(tone, 0, 255).astype(np.uint8)
    return he, float(gap.mean())


def case_figure(arms: dict, rid: str, m: dict, v: dict, corr: dict, *, smooth: int = 5,
                max_dist: float = 13.0, dot_scale: float = 0.65, scalebar: float = 200.0):
    """The published five-panel figure: H&E, both arms' cell classes, both arms' neighbourhoods."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from sklearn.neighbors import NearestNeighbors

    NC = m["NC"]
    k = arms["rids"].index(rid)
    mR = np.where(m["reg"] == k)[0]
    mV = np.where(v["reg"] == k)[0]
    he, gap = fill_gaps(arms["he"])
    ds = arms["ds"]
    shape = (he.shape[0] * ds, he.shape[1] * ds)
    max_px = max_dist / MPP
    coverR = raster_voronoi(m["x"][mR], m["y"][mR], np.zeros(len(mR), int),
                            shape, ds, max_px) >= 0
    coverV = raster_voronoi(v["x"][mV], v["y"][mV], np.zeros(len(mV), int),
                            shape, ds, max_px) >= 0
    bg = np.array(matplotlib.colors.to_rgb(BG), np.float32)

    def nb_of(x, y, n):
        if smooth <= 1:
            return None
        return NearestNeighbors(n_neighbors=min(smooth, n)).fit(
            np.c_[x, y]).kneighbors(np.c_[x, y])[1]

    fig = plt.figure(figsize=(11.0, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1], left=0.008, right=0.992,
                          top=0.890, bottom=0.185, wspace=0.028, hspace=0.115)
    ah = fig.add_subplot(gs[:, 0])
    ah.imshow(np.clip(np.asarray(he, np.float32) / 255.0, 0, 1), interpolation="antialiased")
    ah.set_title(rid, fontsize=9)

    for j, (keys, arm, sel, ttl) in enumerate([
            (corr["key_m"], m, mR, "cell types, REAL MIF segmentation"),
            (corr["key_v"], v, mV, "cell types, VIRTUAL MIF segmentation")]):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.set_facecolor("#ffffff")
        F = keys[sel]
        dsz = dot_scale * float(np.clip(22000.0 / max(len(F), 1), 0.8, 4.5))
        share = {f: float(np.mean(F == f)) for f in set(F)}
        for f in sorted(set(F), key=lambda q: -share[q]):     # commonest first, so it goes under
            s_ = F == f
            ax.scatter(arm["x"][sel][s_] / ds, arm["y"][sel][s_] / ds, s=dsz,
                       c=corr["colour"][f], linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=9)

    for j, (arm, sel, cols, cover, ttl) in enumerate([
            (m, mR, corr["cn_col_m"], coverR, "CN, REAL MIF"),
            (v, mV, corr["cn_col_v"], coverV, "CN, VIRTUAL MIF")]):
        ax = fig.add_subplot(gs[1, j + 1])
        lab = arm["cn"][sel]
        nb = nb_of(arm["x"][sel], arm["y"][sel], len(sel))
        if nb is not None:
            lab = np.array([np.bincount(r_, minlength=NC).argmax() for r_ in lab[nb]])
        pal = (np.array([matplotlib.colors.to_rgb(q) for q in [BG] + list(cols)])
               * 255).astype(np.uint8)
        Vg = raster_voronoi(arm["x"][sel], arm["y"][sel], lab, shape, ds, max_px)
        ax.imshow(np.where(cover[..., None], pal[np.clip(Vg + 1, 0, len(pal) - 1)] / 255.0, bg),
                  interpolation="antialiased")
        ax.set_title(ttl, fontsize=9)

    ry, rx = np.where(coverR | coverV)
    pad = 0.02 * max(coverR.shape)
    x0, x1, y0, y1 = rx.min() - pad, rx.max() + pad, ry.min() - pad, ry.max() + pad
    for ax in fig.axes:
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
    bar = scalebar / MPP / ds
    bx, by = x0 + 0.05 * (x1 - x0), y1 - 0.045 * (y1 - y0)
    ah.add_patch(Rectangle((bx, by), bar, max((y1 - y0) * 0.009, 2), fc="k", ec="none",
                           clip_on=False))
    ah.text(bx + bar / 2, by - 0.012 * (y1 - y0), f"{scalebar:g} µm", ha="center", va="bottom",
            fontsize=7)

    # Names only. The shares, the correspondence tier and the method all live in the notebook
    # around the figure; a legend that repeats them makes the panel harder to read, not more
    # complete.
    cats = sorted(corr["categories"],
                  key=lambda q: -(np.mean(corr["key_m"][mR] == q)
                                  + np.mean(corr["key_v"][mV] == q)))
    lg = [Line2D([], [], marker="o", ls="none", ms=5.5, mfc=corr["colour"][q], mec="none",
                 label=disp(q.replace(" (real only)", "").replace(" (virtual only)", "")))
          for q in cats]
    L = fig.legend(handles=lg, loc="upper center", bbox_to_anchor=(0.5, 0.150), ncol=5,
                   fontsize=8.5, frameon=False)
    L._legend_box.align = "left"
    hs = [Patch(fc=corr["cn_col_m"][c], label=disp(corr["cn_uni_m"][c])) for c in range(NC)]
    L2 = fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 0.085), ncol=3,
                    fontsize=8.5, frameon=False)
    L2._legend_box.align = "left"

    fig.suptitle(rid, fontsize=11, y=0.972)
    return fig, {"gap_fraction": gap, "n_real": int(len(mR)), "n_virtual": int(len(mV))}
