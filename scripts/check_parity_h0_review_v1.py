"""Parity check at h = 0 (review v1): do odd and even chains differ for the random chain?

Exact free fermions, fixed_mean ensemble, L = 127, 128, 129 with seeds 0-999. Per realization,
on the window Gamma0 = 0.85-1.10 (step 0.01): maximum of xi_lambda (closed form, Eq. 6) and of
the physical response -dm2_bulk/dGamma0; at Gamma0 = 1: fraction of nearest-neighbour MST
edges, largest H0 cluster (C_ij >= 0.6), largest SDRG cluster and their adjusted Rand index.

Run:  python scripts/check_parity_h0_review_v1.py
Output: results/review_v1/parity-h0-v1/summary.json
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from clean_tfim_tn.disorder.reentrant import generate_disorder_base  # noqa: E402
from clean_tfim_tn.exact.free_fermion_h0 import cstag_elimination, majorana_state  # noqa: E402
from compare_sdrg_h0_clusters_review_v1 import ari, components, largest, sdrg_partition  # noqa: E402

OUT = ROOT / "results" / "review_v1" / "parity-h0-v1"
GRID = np.round(np.arange(0.85, 1.10 + 1e-9, 0.01), 4)
SIZES = (127, 128, 129)
N = 1000
C0 = 0.6


def peak(x, y):
    i = int(np.argmax(y[1:-1])) + 1
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    return float(x[i] + (0.5 * (a - c) / den * (x[1] - x[0]) if den < 0 else 0.0)), i in (1, len(y) - 2)


def task(args):
    L, seed = args
    base = generate_disorder_base(L, seed)
    deaths, m2 = [], []
    k1 = int(np.flatnonzero(GRID == 1.0)[0])
    for k, g in enumerate(GRID):
        r = base.realize(float(g))
        C = cstag_elimination(majorana_state(r.J, r.Gamma)[0])
        D = np.clip(1.0 - C, 0.0, 2.0) + 1.0
        np.fill_diagonal(D, 0.0)
        T = minimum_spanning_tree(np.triu(D)).tocoo()
        w = T.data - 1.0
        deaths.append(np.sort(w))
        m2.append(C[L // 4:3 * L // 4, L // 4:3 * L // 4].mean())
        if k == k1:
            nn_frac = float(np.mean(np.abs(T.col - T.row) == 1))
            h0 = components(L, [(i, j) for i, j, x in zip(T.row, T.col, w) if x <= 1.0 - C0])
            sd = sdrg_partition(r.J, r.Gamma)
            clus = (len(largest(h0)), len(largest(sd)), ari(sd, h0))
    d = np.array(deaths)
    dd = np.gradient(d, GRID, axis=0, edge_order=2)
    xi = np.sqrt((d * dd ** 2).sum(1) / (2 * (L - 1)))
    phys = -np.gradient(np.array(m2), GRID, edge_order=2)
    return L, peak(GRID, xi), peak(GRID, phys), nn_frac, clus


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=12) as pool:
        res = list(pool.map(task, [(L, s) for L in SIZES for s in range(N)], chunksize=4))
    summary = {}
    for L in SIZES:
        rows = [r for r in res if r[0] == L]
        xi = np.array([r[1][0] for r in rows if not r[1][1]])
        ph = np.array([r[2][0] for r in rows if not r[2][1]])
        cl = np.array([r[4] for r in rows])
        summary[f"L{L}"] = {"xi_peak_mean": float(xi.mean()), "xi_peak_std": float(xi.std(ddof=1)), "xi_edge": len(rows) - xi.size,
                            "phys_peak_mean": float(ph.mean()), "phys_peak_std": float(ph.std(ddof=1)),
                            "mst_nn_fraction_at_1": float(np.mean([r[3] for r in rows])),
                            "largest_H0_c0.6": float(cl[:, 0].mean()), "largest_SDRG": float(cl[:, 1].mean()), "ARI": float(cl[:, 2].mean()),
                            "sem_xi_mean": float(xi.std(ddof=1) / np.sqrt(xi.size))}
        print(f"L{L}: " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in summary[f"L{L}"].items()), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
