"""Crossing analysis of the H0 cluster mass at the IDFP (review v1, option 3, refined grid).

Input: results/review_v1/free-fermion-mst-clusters-v2/random_L{L}.npz (Gamma0 = 0.90-1.10).
For each threshold c0 the largest H0 cluster at scale 1 - c0 (components of C_ij >= c0, from
the MST) has mass S. With the exact IDFP fractal dimension d_f = (1 + sqrt 5)/4, the scaled
mass Q_L(Gamma0) = <S> / L^{d_f} is size independent at Gamma_c = 1. We locate the crossing
of Q_L and Q_{2L} for consecutive sizes (linear interpolation of the difference on the grid),
with a realization bootstrap, and compare with
  NN      largest contiguous run with C_{i,i+1} >= c0 (nearest-neighbour clusters)
  m2      m2_bulk * L^{2x},  x = 1 - d_f   (bulk mean correlation)
  Chalf   C_half * L^{2x}                  (mean correlation at distance L/2)
Local fractal dimension at Gamma0 = 1 between consecutive sizes is also reported.

Run:  python scripts/analyze_free_fermion_mst_clusters_v2_review_v1.py
Output: results/review_v1/free-fermion-mst-clusters-v2/analysis/summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_free_fermion_mst_clusters_review_v1 import clusters, nn_run  # noqa: E402

DATA = ROOT / "results" / "review_v1" / "free-fermion-mst-clusters-v2"
OUT = DATA / "analysis"
THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8)
DF = (1 + 5 ** 0.5) / 4
X = 1 - DF
N_BOOT = 500
RNG = np.random.default_rng(20261018)


def crossing(g, a, b):
    d = b - a
    s = np.flatnonzero(np.sign(d[:-1]) != np.sign(d[1:]))
    if s.size == 0:
        return np.nan
    i = s[np.argmin(np.abs(g[s] - 1.0))]
    return float(g[i] - d[i] * (g[i + 1] - g[i]) / (d[i + 1] - d[i]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = {}
    for f in sorted(DATA.glob("random_L*.npz"), key=lambda p: int(p.stem.split("L")[1])):
        z = np.load(f)
        L, g = int(z["L"]), np.asarray(z["Gamma0"], float)
        ends, w, cnn = z["mst_ends"], z["mst_w"], z["cnn"]
        N, G = w.shape[:2]
        obs = {"m2": z["m2_bulk"] * L ** (2 * X), "Chalf": z["C_half"] * L ** (2 * X)}
        for c0 in THRESHOLDS:
            S = np.zeros((N, G))
            Snn = np.zeros((N, G))
            for n in range(N):
                for k in range(G):
                    S[n, k] = clusters(L, ends[n, k], w[n, k], c0)[0]
                    Snn[n, k] = nn_run(cnn[n, k], c0)
            obs[f"S{c0}"] = S / L ** DF
            obs[f"NN{c0}"] = Snn / L ** DF
        per[L] = {"g": g, "obs": obs}
        print(f"L{L} done", flush=True)
    sizes = sorted(per)
    summary = {"d_f": DF, "thresholds": THRESHOLDS, "crossings": {}, "Q_at_1": {}, "local_df_at_1": {}}
    for key in per[sizes[0]]["obs"]:
        rows = []
        for La, Lb in zip(sizes[:-1], sizes[1:]):
            g = np.intersect1d(np.round(per[La]["g"], 6), np.round(per[Lb]["g"], 6))   # common field values
            ia = np.isin(np.round(per[La]["g"], 6), g)
            ib = np.isin(np.round(per[Lb]["g"], 6), g)
            A, B = per[La]["obs"][key][:, ia], per[Lb]["obs"][key][:, ib]
            x0 = crossing(g, A.mean(0), B.mean(0))
            boot = [crossing(g, A[RNG.integers(0, len(A), len(A))].mean(0), B[RNG.integers(0, len(B), len(B))].mean(0))
                    for _ in range(N_BOOT)]
            boot = np.array(boot)
            rows.append({"L": [La, Lb], "crossing": x0, "ci95": np.nanpercentile(boot, [2.5, 97.5]).tolist() if np.isfinite(boot).any() else None,
                         "missing_fraction": float(np.mean(~np.isfinite(boot)))})
        summary["crossings"][key] = rows
        k1 = {L: int(np.argmin(np.abs(per[L]["g"] - 1.0))) for L in sizes}
        summary["Q_at_1"][key] = {L: float(per[L]["obs"][key][:, k1[L]].mean()) for L in sizes}
        q = [summary["Q_at_1"][key][L] for L in sizes]
        summary["local_df_at_1"][key] = [float(np.log(q[i + 1] / q[i]) / np.log(2)) for i in range(len(q) - 1)]
        print(f"{key:7s} crossings " + "  ".join(f"{r['L'][0]}-{r['L'][1]}: {r['crossing']:.4f}" for r in rows)
              + " | Q(1) " + " ".join(f"{v:.3f}" for v in q), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
