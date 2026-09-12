from __future__ import annotations

import numpy as np


def aggregate(values: np.ndarray, groups: np.ndarray, thresh: float | None = None) -> dict:
    """Summarise per-patch values within each group (region or patient)."""
    v = np.asarray(values, np.float64)
    if v.ndim == 1:
        v = v[:, None]
    ids = np.array(sorted(set(np.asarray(groups).tolist())), dtype=object)
    out = {k: [] for k in ("mean", "p75", "p95", "frac_above", "n_patches")}
    for g in ids:
        m = np.asarray(groups) == g
        sub = v[m]
        out["mean"].append(sub.mean(0))
        out["p75"].append(np.percentile(sub, 75, axis=0))
        out["p95"].append(np.percentile(sub, 95, axis=0))
        out["frac_above"].append((sub > thresh).mean(0) if thresh is not None
                                 else np.full(v.shape[1], np.nan))
        out["n_patches"].append(m.sum())
    return {"ids": ids, **{k: np.array(val) for k, val in out.items()}}


def patient_folds(patients: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Patient-disjoint fold ids, balanced by patch count."""
    p = np.asarray(patients)
    rng = np.random.default_rng(seed)
    uniq = sorted(set(p.tolist()))
    rng.shuffle(uniq)
    uniq.sort(key=lambda u: -(p == u).sum())
    folds = np.full(len(p), -1)
    load = np.zeros(k)
    for u in uniq:
        f = int(load.argmin())
        m = p == u
        folds[m] = f
        load[f] += m.sum()
    return folds


def _fit_predict(X, y, folds, seed=0):
    """Out-of-fold ridge-regularised logistic probe; returns OOF scores."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes = sorted(set(y.tolist()))
    yi = np.array([classes.index(v) for v in y])
    oof = np.zeros(len(yi))
    torch.manual_seed(seed)
    for f in sorted(set(folds.tolist())):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32, device=dev)
        Xte = torch.tensor((X[te] - mu) / sd, dtype=torch.float32, device=dev)
        ytr = torch.tensor(yi[tr], dtype=torch.long, device=dev)
        w = torch.tensor([len(ytr) / (len(classes) * max((yi[tr] == c).sum(), 1))
                          for c in range(len(classes))], dtype=torch.float32, device=dev)
        lin = torch.nn.Linear(X.shape[1], len(classes)).to(dev)
        opt = torch.optim.Adam(lin.parameters(), lr=0.05, weight_decay=1.0 / len(tr))
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        for _ in range(300):
            opt.zero_grad()
            lossf(lin(Xtr), ytr).backward()
            opt.step()
        with torch.no_grad():
            oof[te] = torch.softmax(lin(Xte), 1)[:, -1].cpu().numpy()
    return oof, yi


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank AUC of a score against a binary label."""
    label = np.asarray(label).astype(int)
    if len(set(label.tolist())) < 2:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def contribution(X: np.ndarray, y: np.ndarray, feature_names: list[str], groups: np.ndarray,
                 k: int = 5, seed: int = 0) -> list[dict]:
    """Leave-one-feature-out drop in out-of-fold AUC, per feature."""
    folds = patient_folds(groups, k=k, seed=seed)
    oof, yi = _fit_predict(X, y, folds, seed=seed)
    full = auc(oof, yi)
    rows = [{"feature": "— full panel —", "auc": full, "drop": 0.0}]
    for j, name in enumerate(feature_names):
        keep = [i for i in range(X.shape[1]) if i != j]
        o, _ = _fit_predict(X[:, keep], y, folds, seed=seed)
        a = auc(o, yi)
        rows.append({"feature": name, "auc": a, "drop": full - a})
    rows[1:] = sorted(rows[1:], key=lambda r: -r["drop"])
    return rows
