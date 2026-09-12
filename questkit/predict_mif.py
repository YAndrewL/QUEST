from __future__ import annotations

import numpy as np

METHODS = ("pearson", "spearman")


def predict_region(gen, he8: np.ndarray, panel: list[str], *, batch: int = 16) -> np.ndarray:
    """A whole region, patch by patch -> (len(panel), H, W) float [0,1]."""
    from quest import PATCH

    he8 = np.asarray(he8)
    if he8.ndim != 3 or he8.shape[2] != 3:
        raise ValueError(f"expected (H,W,3) H&E, got {he8.shape}")
    he = he8.astype(np.float32) / 255.0 if he8.dtype == np.uint8 else np.asarray(he8, np.float32)
    H, W = he.shape[:2]
    if H < PATCH or W < PATCH:
        raise ValueError(f"region {H}x{W} is smaller than one {PATCH}px patch")
    ys = list(range(0, H - PATCH + 1, PATCH)) + ([H - PATCH] if H % PATCH else [])
    xs = list(range(0, W - PATCH + 1, PATCH)) + ([W - PATCH] if W % PATCH else [])
    corners = [(y, x) for y in ys for x in xs]

    out = np.zeros((len(panel), H, W), np.float32)
    cov = np.zeros((H, W), np.float32)
    for s in range(0, len(corners), batch):
        blk = corners[s:s + batch]
        pred = gen(np.stack([he[y:y + PATCH, x:x + PATCH] for y, x in blk]), panel, batch=batch)
        for (y, x), p in zip(blk, pred):
            out[:, y:y + PATCH, x:x + PATCH] += p
            cov[y:y + PATCH, x:x + PATCH] += 1
    return out / np.maximum(cov, 1)[None]


def grid_pool(x: np.ndarray, g: int) -> np.ndarray:
    """(N,H,W) -> (N, g*g) by mean-pooling each image into a g x g array of cells."""
    x = np.asarray(x, np.float32)
    n, h, w = x.shape
    if h % g or w % g:
        raise ValueError(f"grid {g} does not divide {h}x{w}")
    return x.reshape(n, g, h // g, g, w // g).mean(axis=(2, 4)).reshape(n, g * g)


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks along axis 1 -- ties share a rank, which background zeros need."""
    from scipy.stats import rankdata
    return rankdata(x, method="average", axis=1).astype(np.float32)


def per_patch_corr(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> np.ndarray:
    """Correlation of each row of two (N,D) arrays -> (N,), NaN where a row has no variance."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    a = np.array(a, np.float32)
    b = np.array(b, np.float32)
    if method == "spearman":
        a, b = _ranks(a), _ranks(b)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    r = np.full(len(a), np.nan)
    ok = den > 1e-6
    r[ok] = num[ok] / den[ok]
    return r


def fidelity(pred: np.ndarray, real: np.ndarray, panel: list[str],
             grid_sizes: tuple[int, ...] = (16,)) -> dict:
    """Per-marker fidelity of (N,C,H,W) predictions against (N,C,H,W) measurements."""
    pred, real = np.asarray(pred, np.float32), np.asarray(real, np.float32)
    if pred.shape != real.shape:
        raise ValueError(f"shape mismatch: predicted {pred.shape} vs real {real.shape}")
    if pred.shape[1] != len(panel):
        raise ValueError(f"{pred.shape[1]} channels but {len(panel)} marker names")
    n = pred.shape[0]

    levels = {"pixel": None}
    levels.update({f"grid{g}": g for g in grid_sizes})
    res = {"panel": list(panel), "n_patches": int(n)}
    for level, g in levels.items():
        res[level] = {m: {k: np.empty(len(panel)) for k in ("mean", "median", "n_valid")}
                      for m in METHODS}
        for i in range(len(panel)):
            p, q = pred[:, i], real[:, i]
            if g is None:
                p, q = p.reshape(n, -1), q.reshape(n, -1)
            else:
                p, q = grid_pool(p, g), grid_pool(q, g)
            for m in METHODS:
                r = per_patch_corr(p, q, m)
                ok = np.isfinite(r)
                res[level][m]["mean"][i] = r[ok].mean() if ok.any() else np.nan
                res[level][m]["median"][i] = np.median(r[ok]) if ok.any() else np.nan
                res[level][m]["n_valid"][i] = int(ok.sum())
    return res


def case_figure(he: np.ndarray, arms, panel: list[str], *, grid: int = 16, scale: float = 1.0):
    """The fidelity figure for ONE patch: H&E, then one row per model arm, one column per marker."""
    import matplotlib.pyplot as plt

    from . import plotting

    arms = list(arms)
    for label, images, res in arms:
        if len(images) != len(panel):
            raise ValueError(f"{label}: {len(images)} channels for {len(panel)} markers")
        if res is not None and list(res["panel"]) != list(panel):
            raise ValueError(f"{label}: res was computed on a different panel")

    n, rows = len(panel), len(arms)
    fig, axes = plt.subplots(rows, n + 1, figsize=(scale * (n + 1), scale * rows + 0.75),
                             squeeze=False)

    axes[0, 0].imshow(he)
    plotting.bare(axes[0, 0])
    axes[0, 0].set_title("H&E", fontsize=8.5, color=plotting.INK)
    for r in range(1, rows):
        axes[r, 0].axis("off")
    axes[rows - 1, 0].text(0.5, 0.5, f"pixel   224 x 224 px\ngrid{grid}  {grid} x {grid} pixels",
                           transform=axes[rows - 1, 0].transAxes, ha="center", va="center",
                           fontsize=6.5, color=plotting.MUTED, linespacing=1.6)

    for r, (label, images, res) in enumerate(arms):
        for j, m in enumerate(panel):
            ax = axes[r, j + 1]
            plotting.show_marker(ax, images[j])
            if r == 0:
                ax.set_title(m, fontsize=8.5, color=plotting.INK)
            if res is not None:
                ax.set_xlabel(f"r {res['pixel']['pearson']['mean'][j]:+.2f} "
                              f"$\\rho$ {res['pixel']['spearman']['mean'][j]:+.2f}\n"
                              f"r {res[f'grid{grid}']['pearson']['mean'][j]:+.2f} "
                              f"$\\rho$ {res[f'grid{grid}']['spearman']['mean'][j]:+.2f}",
                              fontsize=7.0, color=plotting.INK, linespacing=1.45)
            if j == 0:
                ax.set_ylabel(label, fontsize=6.5, color=plotting.MUTED)
    fig.tight_layout(w_pad=0.3, h_pad=0.4)
    return fig
