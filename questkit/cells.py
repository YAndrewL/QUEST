from __future__ import annotations

import numpy as np


def cell_means(stack: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(C,H,W) stack + label image -> (n_cells, C) means and the cell ids they belong to."""
    stack = np.asarray(stack, np.float32)
    if stack.ndim == 2:
        stack = stack[None]
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return np.zeros((0, stack.shape[0]), np.float32), ids
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    sflat = flat[order]
    s = np.searchsorted(sflat, ids, "left")
    e = np.searchsorted(sflat, ids, "right")
    gf = stack.reshape(len(stack), -1)[:, order]
    means = np.stack([gf[:, a:b].mean(1) for a, b in zip(s, e)]).astype(np.float32)
    return means, ids


def centroids(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cell centroids in pixels, ascending label id -> (x, y)."""
    ids = np.unique(labels)
    ids = ids[ids > 0]
    h, w = labels.shape
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    sflat = flat[order]
    s = np.searchsorted(sflat, ids, "left")
    e = np.searchsorted(sflat, ids, "right")
    xs = np.empty(len(ids), np.float64)
    ys = np.empty(len(ids), np.float64)
    for i, (a, b) in enumerate(zip(s, e)):
        px = order[a:b]
        ys[i] = (px // w).mean()
        xs[i] = (px % w).mean()
    return xs, ys


def scale_p99(v: np.ndarray) -> np.ndarray:
    """Per-channel robust scaling to [0,1] by the 99th percentile."""
    v = np.asarray(v, np.float32)
    hi = np.percentile(v, 99)
    return np.clip(v / hi, 0, 1) if hi > 0 else np.zeros_like(v)
