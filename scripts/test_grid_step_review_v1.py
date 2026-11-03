"""Referee 2 (2.3 and 5): sensitivity of the maxima of xi_lambda to the field step, exact h = 0 data.

The exact free-fermion scans (results/review_v1/free-fermion-h0-v1, step 0.005) are subsampled to
steps 0.01 and 0.02, and the maximum of xi_lambda [Eq. (6), central differences, three-point
parabola] is recomputed. Clean chain L = 16-512, random chain L = 16-256 (all realizations):
shift of each maximum with respect to the original step, and change of the width of the
distribution of the maxima.
Run:  python scripts/test_grid_step_review_v1.py
Output: results/review_v1/grid-step-v1/summary.json
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "review_v1" / "free-fermion-h0-v1"
OUT = ROOT / "results" / "review_v1" / "grid-step-v1"
STRIDES = (1, 2, 4)


def xi(g, d):
    dd = np.gradient(d, g, axis=-2, edge_order=2)
    return np.sqrt((d * dd ** 2).sum(-1) / (2 * d.shape[-1]))


def peak(g, y):
    i = int(np.argmax(y[1:-1])) + 1
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    return float(g[i] + (0.5 * (a - c) / den * (g[1] - g[0]) if den < 0 else 0.0))


def peaks(g, deaths, stride):
    gs, ds = g[::stride], deaths[..., ::stride, :]
    if ds.ndim == 2:
        return np.array([peak(gs, xi(gs, ds))])
    return np.array([peak(gs, xi(gs, d)) for d in ds])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"strides": STRIDES}
    for kind, sizes in (("clean", (16, 32, 64, 128, 256, 512)), ("random", (16, 32, 64, 128, 256))):
        for L in sizes:
            with np.load(DATA / f"{kind}_L{L}.npz") as z:
                g, deaths = np.asarray(z["Gamma0"], float), np.asarray(z["deaths"], float)
            ref = peaks(g, deaths, 1)
            row = {"step": float(g[1] - g[0]), "n": int(ref.size)}
            for s in STRIDES[1:]:
                p = peaks(g, deaths, s)
                d = np.abs(p - ref)
                row[f"stride{s}"] = {"step": float(s * (g[1] - g[0])), "max_shift": float(d.max()),
                                     "median_shift": float(np.median(d)), "mean_shift": float(d.mean())}
                if ref.size > 1:
                    row[f"stride{s}"]["width_ratio"] = float(p.std(ddof=1) / ref.std(ddof=1))
                    row[f"stride{s}"]["frac_shift_gt_0.025"] = float(np.mean(d > 0.025))
            if ref.size > 1:
                row["width"] = float(ref.std(ddof=1))
            out[f"{kind}_L{L}"] = row
            print(kind, L, json.dumps(row), flush=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
