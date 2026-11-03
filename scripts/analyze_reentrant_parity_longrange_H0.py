"""Parity-resolved long-range H0 level as an exit detector (OBC L16 / L32).

For each reentrant seed and Gamma0, the maximum spanning forest of C=Cstag
(equivalently the H0 single-linkage forest of D=1-C) is computed on three
edge sets: P0 = pairs at distance exactly r=L/2-1 (exact identity with the
physical correlator), P1 = odd distances >= r-2, P2 = all distances >= r-2.
The level M = mean of the selected C values; the exit estimate is its last
downward zero crossing after the physical entry. Exploratory, no DMRG.

Output: reentrant-exit-anatomy-v1/analysis/parity_longrange_H0.json
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "random_ising_gamma_ensemble" / "reentrant-exit-anatomy-v1" / "analysis"


def msf_level(C, edges):
    L = C.shape[0]
    w = np.array([C[i, j] for i, j in edges])
    parent = list(range(L))
    chosen = []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for k in np.argsort(-w, kind="stable"):
        i, j = edges[k]
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            chosen.append(w[k])
    return float(np.mean(chosen))


def last_down(g, y, after):
    idx = np.flatnonzero((y[:-1] > 0) & (y[1:] <= 0) & (g[:-1] >= after))
    if not idx.size:
        return np.nan
    i = idx[-1]
    return g[i] + (g[i + 1] - g[i]) * y[i] / (y[i] - y[i + 1])


def main():
    summary = {}
    for L in (16, 32):
        z = np.load(OUT / f"cache_L{L}.npz")
        g = z["Gamma0"]
        step = g[1] - g[0]
        sel = np.flatnonzero(z["selected"])
        C = z["Cstag"][sel].astype(float)
        R = L // 2 - 1
        s = R - 2
        pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]
        variants = {"P0_distance_r": [p for p in pairs if p[1] - p[0] == R],
                    "P1_odd_ge_r_minus_2": [p for p in pairs if (p[1] - p[0]) % 2 == 1 and p[1] - p[0] >= s],
                    "P2_all_ge_r_minus_2": [p for p in pairs if p[1] - p[0] >= s]}
        entry, exit_ = z["entry"][sel], z["exit"][sel]
        res = {"const_median_MAE": float(np.mean(np.abs(np.median(exit_) - exit_)))}
        for name, edges in variants.items():
            est = np.array([last_down(g, np.array([msf_level(C[n, i], edges) for i in range(len(g))]), entry[n])
                            for n in range(sel.size)])
            ok = np.isfinite(est)
            row = {"coverage": int(ok.sum()), "selected": int(sel.size)}
            if ok.sum() >= 3:
                e, r = est[ok], exit_[ok]
                row.update({"MAE": float(np.mean(np.abs(e - r))), "one_step": float(np.mean(np.abs(e - r) <= step + 1e-9)),
                            "spearman": float(spearmanr(e, r).statistic), "bias": float(np.mean(e - r))})
            res[name] = row
        summary[f"L{L}"] = res
    (OUT / "parity_longrange_H0.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
