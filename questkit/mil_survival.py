from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "crc-metu"
PANEL8 = ["DAPI", "CD3e", "CD8", "CD20", "CD68", "Ki67", "PDL1", "PanCK"]


# ---------------------------------------------------------------------------------------------
# data


def load_bags(root: Path = DATA, panel: list[str] | None = None,
              endpoint: str = "survival_metU", event_col: str = "os_event_metU",
              event_pos: str = "1") -> tuple[list[dict], list[tuple]]:
    """One bag per staged slide: its virtual-MIF patch features, its patient, its label."""
    panel = panel or PANEL8
    index = json.loads((root / "slide_index.json").read_text())
    lab = {r["patient_id"]: r for r in csv.DictReader(open(root / "labels.csv"))}
    bags, ys = [], []
    for si in index:
        f = root / "feats" / f"slide{si['slide_idx']:04d}.npz"
        if not f.exists():
            continue
        with np.load(f, allow_pickle=True) as z:
            arr = np.asarray(z["dino_vmif"], np.float32)
            have = [str(m) for m in z["panel"]]
        idx = [have.index(m) for m in panel if m in have]
        if not idx:
            continue

        L = lab.get(si["patient_id"], {})
        t, e = str(L.get(endpoint, "")).strip(), str(L.get(event_col, "")).strip()
        if t in ("", "nan", "None") or e in ("", "nan", "None"):
            continue
        bags.append({"pid": si["patient_id"],
                     "x": arr[:, idx, :].reshape(arr.shape[0], -1)})
        ys.append((float(t), 1 if e == event_pos else 0))
    return bags, ys


# ---------------------------------------------------------------------------------------------
# survival helpers


def cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Cox partial likelihood. Sorted by descending time, so the risk set of i is [0..i]."""
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    return -((risk - log_cumsum) * event).sum() / event.sum().clamp_min(1.0)


def c_index(time: np.ndarray, risk: np.ndarray, event: np.ndarray) -> float | None:
    """Harrell's C. A pair (i, j) is comparable when i had an event and t_i < t_j."""
    num = den = 0.0
    for i in range(len(time)):
        if event[i] != 1:
            continue
        m = time > time[i]
        den += m.sum()
        num += (risk[i] > risk[m]).sum() + 0.5 * (risk[i] == risk[m]).sum()
    return None if den == 0 else float(num / den)


def stratified_group_kfold(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    """Whole groups to folds, balancing the per-class counts. No group spans two folds."""
    y = np.asarray(y)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    labels = np.unique(y)
    gc = {g: np.array([(y[groups == g] == c).sum() for c in labels]) for g in uniq}
    uniq = sorted(uniq, key=lambda g: -gc[g].sum())
    fold_counts = np.zeros((n_splits, len(labels)))
    group_fold: dict = {}
    for g in uniq:
        costs = []
        for f in range(n_splits):
            trial = fold_counts.copy()
            trial[f] += gc[g]
            costs.append(np.std(trial, axis=0).sum())
        f = int(np.argmin(costs))
        group_fold[g] = f
        fold_counts[f] += gc[g]
    for f in range(n_splits):
        va = np.array([group_fold[g] == f for g in groups])
        yield np.where(~va)[0], np.where(va)[0]


# ---------------------------------------------------------------------------------------------
# model


class Stream(nn.Module):
    """Aggregate one patch's feature vector to d. This is the MARKER-axis aggregation."""

    def __init__(self, in_dim: int, d: int, markers: bool = False, dropout: float = 0.3):
        super().__init__()
        self.markers = markers
        if markers:
            self.mq = nn.Parameter(torch.randn(768) * 0.02)
            in_dim = 768
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d), nn.GELU(),
                                  nn.Dropout(dropout))

    def forward(self, x):
        if self.markers:
            s = torch.softmax(x @ self.mq, 1).unsqueeze(-1)
            x = (s * x).sum(1)
        return self.proj(x)


class GatedAttnPool(nn.Module):
    """Gated attention over the patches of a bag. This is the PATCH-axis aggregation."""

    def __init__(self, d: int, attn: int = 96):
        super().__init__()
        self.V = nn.Linear(d, attn)
        self.U = nn.Linear(d, attn)
        self.w = nn.Linear(attn, 1)

    def forward(self, h):                                  # (T, d) -> (d,)
        a = torch.softmax(self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))), 0)
        return (a * h).sum(0)


class MIL(nn.Module):
    """One bag of patch vectors, pooled by gated attention, into a scalar risk."""

    def __init__(self, in_dim: int, d: int = 192, markers: bool = False, dropout: float = 0.3,
                 head: str = "mlp"):
        super().__init__()
        self.stream = Stream(in_dim, d, markers=markers, dropout=dropout)
        self.pool = GatedAttnPool(d)
        self.head = (nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)) if head == "linear"
                     else nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(128, 1)))

    def forward(self, x):
        return self.head(self.pool(self.stream(x)))


# ---------------------------------------------------------------------------------------------
# fit


def oof(bags, ys, device, folds=5, epochs=40, lr=3e-4, seed=0, head="mlp", markers=False,
        loo=True):
    """Out-of-fold risk, and the per-patch leave-one-out contribution."""
    dim = bags[0]["x"].shape[1]
    ev = np.array([y[1] for y in ys])
    grp = np.array([b["pid"] for b in bags])
    risk = np.zeros(len(bags))
    contrib: list = [None] * len(bags)

    for tr, va in stratified_group_kfold(ev, grp, folds, seed):
        cat = np.concatenate([bags[i]["x"] for i in tr], 0)
        mu = cat.mean(0, keepdims=True).astype(np.float32)
        sd = cat.std(0, keepdims=True).astype(np.float32) + 1e-6
        prep = lambda b: torch.from_numpy(((b["x"] - mu) / sd).astype(np.float32)).to(device)
        Xtr = [prep(bags[i]) for i in tr]
        Xva = [prep(bags[i]) for i in va]

        torch.manual_seed(seed)
        model = MIL(dim, markers=markers, head=head).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        t = torch.tensor([ys[i][0] for i in tr], dtype=torch.float32, device=device)
        e = torch.tensor([ys[i][1] for i in tr], dtype=torch.float32, device=device)
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            r = torch.stack([model(x) for x in Xtr]).squeeze(1)
            cox_loss(r, t, e).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            for j, x in zip(va, Xva):
                risk[j] = model(x).item()
                if loo:
                    d = np.empty(x.shape[0], np.float32)
                    for i in range(x.shape[0]):
                        d[i] = risk[j] - model(torch.cat([x[:i], x[i + 1:]], 0)).item()
                    contrib[j] = d.tolist()
    return risk, contrib


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Patient-level survival MIL on CRC-metU's virtual-MIF features.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DATA), help="staged cohort (default: data/crc-metu)")
    ap.add_argument("--panel", default=",".join(PANEL8))
    ap.add_argument("--endpoint", default="survival_metU")
    ap.add_argument("--event-col", default="os_event_metU")
    ap.add_argument("--event-pos", default="1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", choices=["mlp", "linear"], default="mlp")
    ap.add_argument("--markers", action="store_true",
                    help="learned attention over the 8 markers instead of the 6,144-d projection")
    ap.add_argument("--no-loo", action="store_true", help="skip the leave-one-out contribution")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None,
                    help="output JSON; default <root>/mil_oof_<endpoint>_rerun.json, so the "
                         "shipped mil_oof_metU.json is never overwritten")
    a = ap.parse_args()

    root = Path(a.root)
    panel = [q.strip() for q in a.panel.split(",") if q.strip()]
    bags, ys = load_bags(root, panel, a.endpoint, a.event_col, a.event_pos)
    if not bags:
        raise SystemExit(f"no bags under {root} -- see Tutorial 0 "
                         f"(tutorials/00_data.ipynb) for the download")
    n_ev = len({b["pid"] for b, y in zip(bags, ys) if y[1]})
    print(f"read  {root}")
    print(f"      {len(bags)} slides, {len({b['pid'] for b in bags})} patients, {n_ev} deaths, "
          f"{sum(b['x'].shape[0] for b in bags):,} patches")
    print(f"      virtual MIF {bags[0]['x'].shape[1]}-d per patch "
          f"({len(panel)} markers x 768){'  [marker attention]' if a.markers else ''}")

    t = np.array([y[0] for y in ys])
    e = np.array([y[1] for y in ys])
    risk, contrib = oof(bags, ys, a.device, a.folds, a.epochs, a.lr, a.seed,
                              head=a.head, markers=a.markers, loo=not a.no_loo)
    # The readout is per PATIENT, not per slide. Four patients contribute more than one slide, and
    # all four died, so scoring the rows would weight those patients twice or three times and would
    # count one death as several -- which is where a slide-level 62 rows / 42 event rows comes from
    # against the cohort's 57 patients and 37 deaths.
    by_pid = {}
    for i, b_ in enumerate(bags):
        by_pid.setdefault(b_["pid"], []).append(i)
    pids = list(by_pid)
    pt = np.array([t[by_pid[q][0]] for q in pids])
    pe = np.array([e[by_pid[q][0]] for q in pids])
    pr = np.array([risk[by_pid[q]].mean() for q in pids])
    ci = c_index(pt, pr, pe)
    print(f"      virtual-MIF arm, out-of-fold C-index {ci:.3f}   "
          f"({len(pids)} patients, {int(pe.sum())} deaths)")

    out = {"endpoint": a.endpoint, "model": "quest-semantic", "arm": "virt", "unit": "patient",
           "n": len(pids), "events": int(pe.sum()), "n_slides": len(bags),
           "folds": a.folds, "seed": a.seed, "head": a.head,
           "markers": bool(a.markers), "cindex_virt": float(ci), "patients": {}}
    # risk is the mean over a patient's slides; the per-patch arrays belong to the first of them,
    # which is the whole patient for the 53 that have only one slide
    for q in pids:
        i = by_pid[q][0]
        out["patients"][q] = {
            "time": ys[i][0], "event": int(ys[i][1]), "n_slides": len(by_pid[q]),
            "n_tiles": int(bags[i]["x"].shape[0]),
            "risk_virt": float(risk[by_pid[q]].mean()), "loo_virt": contrib[i]}

    dst = Path(a.out) if a.out else root / f"mil_oof_{a.endpoint.split('_')[-1]}_rerun.json"
    dst.write_text(json.dumps(out))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
