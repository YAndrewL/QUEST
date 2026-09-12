from __future__ import annotations

import numpy as np


def fit_typer(X: np.ndarray, y: np.ndarray, panel: list[str], *, steps: int = 300,
              lr: float = 0.05, seed: int = 0) -> dict:
    """(n_cells, n_markers) means + per-cell class names -> a typer dict."""
    import torch

    keep = np.array([v is not None for v in y])
    X, y = np.asarray(X, np.float64)[keep], np.asarray(y, object)[keep]
    classes = sorted({str(v) for v in y})
    ci = {c: i for i, c in enumerate(classes)}
    yi = np.array([ci[str(v)] for v in y])

    mu, sd = X.mean(0), X.std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(yi, dtype=torch.long, device=dev)
    w = torch.tensor([len(yi) / (len(classes) * max((yi == c).sum(), 1))
                      for c in range(len(classes))], dtype=torch.float32, device=dev)
    lin = torch.nn.Linear(X.shape[1], len(classes)).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=1.0 / len(yi))
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    for _ in range(steps):
        opt.zero_grad()
        lossf(lin(Xt), yt).backward()
        opt.step()
    return {"W": lin.weight.detach().cpu().numpy().astype(np.float32),
            "b": lin.bias.detach().cpu().numpy().astype(np.float32),
            "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
            "classes": list(classes), "panel": list(panel), "n_train": int(len(yi))}


def apply_typer(typer: dict, X: np.ndarray, panel: list[str] | None = None) -> np.ndarray:
    """Type every row of `X`. `panel` is checked against the typer's, not reordered silently."""
    if panel is not None and list(panel) != list(typer["panel"]):
        raise ValueError(f"panel mismatch: typer was fitted on {list(typer['panel'])}")
    z = (np.asarray(X, np.float32) - typer["mu"]) / typer["sd"]
    k = (z @ typer["W"].T + typer["b"]).argmax(1)
    return np.array([typer["classes"][i] for i in k], dtype=object)


def save_typers(path, **typers) -> None:
    """Write one or more named typers into a single npz."""
    flat = {}
    for name, t in typers.items():
        for k, v in t.items():
            flat[f"{name}__{k}"] = np.asarray(v)
    np.savez_compressed(path, **flat)


def load_typers(path) -> dict:
    """Read back what `save_typers` wrote -> {name: typer}."""
    z = np.load(path, allow_pickle=True)
    out: dict[str, dict] = {}
    for key in z.files:
        name, k = key.split("__", 1)
        v = z[key]
        out.setdefault(name, {})[k] = (v.tolist() if k in ("classes", "panel")
                                       else str(v) if k in ("holdout", "scheme", "model")
                                       else int(v) if k == "n_train" else v)
    return out
