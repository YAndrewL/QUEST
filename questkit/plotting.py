from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#202427", "#5A6067", "#E6E8EA"


DPI = 120
plt.rcParams["figure.dpi"] = DPI  # for savefig to a file; ignored by the inline backend
plt.rcParams["savefig.dpi"] = DPI


def style(ax, xlabel: str = "", ylabel: str = "", title: str = ""):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#C4C8CC")
    ax.spines["bottom"].set_color("#C4C8CC")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.tick_params(labelsize=8, colors=MUTED)
    return ax


def show_marker(ax, img: np.ndarray, title: str = "", cmap: str = "gray"):
    """One marker map on a robust range, greyscale by default."""
    lo, hi = np.percentile(img, [2, 98])
    ax.imshow(img, cmap=cmap, vmin=lo, vmax=max(hi, lo + 1e-6), interpolation="nearest")
    ax.set_title(title, fontsize=9, color=INK)
    bare(ax)
    return ax


def bare(ax):
    """Strip an image axis to the image: no ticks, no frame."""
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


COARSE15_COLORS = {
    "B cells": "#54A24B", "Macrophages/Monocytes": "#F58518", "Adipocytes": "#C7B299",
    "Dendritic cells": "#EECA3B", "T cells": "#4C78A8", "Granulocytes": "#9D755D",
    "NK cells": "#B279A2", "Nerves": "#7B6888", "Plasma cells": "#8CD17D",
    "Smooth muscle": "#79706E", "Stroma": "#BAB0AC", "Tumor cells": "#E45756",
    "Vasculature/Lymphatics": "#FF9DA6", "Other cells": "#D4D4D4",
    "Background": "#F2F2F2",
}


def paint_types(ax, inst, ids, types, *, palette=None, title: str = "",
                background: str = "#FFFFFF", unlabelled: str = "#DDDFE1"):
    """Paint each cell's instance footprint with its type colour."""
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    palette = palette or COARSE15_COLORS
    inst = np.asarray(inst)
    ids = np.asarray(ids)
    lut = np.zeros((int(max(inst.max(), ids.max() if len(ids) else 0)) + 1, 3), np.float32)
    lut[0] = to_rgb(background)
    for i in ids:
        lut[int(i)] = to_rgb(unlabelled)
    for t, col in palette.items():
        m = np.asarray([v == t for v in types])
        if m.any():
            lut[ids[m].astype(int)] = to_rgb(col)
    ax.imshow(lut[inst], interpolation="nearest")
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return [Patch(facecolor=c, label=t) for t, c in palette.items()]


def paint_values(ax, inst, ids, values, *, vmin=None, vmax=None, cmap="viridis",
                 title: str = "", background: str = "#FFFFFF"):
    """Paint each cell's footprint with a continuous per-cell value."""
    import matplotlib as mpl

    inst = np.asarray(inst)
    ids = np.asarray(ids)
    v = np.asarray(values, np.float64)
    lo = float(np.nanmin(v) if vmin is None else vmin)
    hi = float(np.nanmax(v) if vmax is None else vmax)
    norm = mpl.colors.Normalize(lo, hi)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    lut = np.zeros((int(max(inst.max(), ids.max() if len(ids) else 0)) + 1, 3), np.float32)
    lut[0] = mpl.colors.to_rgb(background)
    ok = np.isfinite(v)
    lut[ids[ok].astype(int)] = sm.to_rgba(v[ok])[:, :3]
    ax.imshow(lut[inst], interpolation="nearest")
    ax.set_title(title, fontsize=8, color=INK)
    bare(ax)
    return sm


def two_colour_markers(ax, a, b, *, title: str = "", ca=(0.85, 0.31, 0.62),
                       cb=(0.18, 0.70, 0.77), pct=(2, 99)):
    """Two marker channels as one image: `a` in magenta, `b` in cyan, on black."""
    out = np.zeros(a.shape + (3,), np.float32)
    for img, col in ((a, ca), (b, cb)):
        lo, hi = np.percentile(img, pct)
        z = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        out += z[..., None] * np.asarray(col, np.float32)
    ax.imshow(np.clip(out, 0, 1), interpolation="nearest")
    ax.set_title(title, fontsize=8, color=INK)
    bare(ax)
    return ax
