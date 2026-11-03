"""Weak-link controls for the PBC ED H1 / low-gap association (cohort v2 only).

Exploratory complement to the frozen v2+v3 inference (v3 cases are not on this
machine). Uses the 600 local v2 cases (L=12..18, offsets -0.025/0/+0.025):
1. location of the four-site H1 motif (gap > 1e-8) on the ring vs bond J;
2. H1 coefficient on log gap01 with the frozen controls (bulk_ms2, window
   mean J and Gamma, window position) plus stratum dummies, with and without
   weak-link covariates (window min J, chain min J, weakest odd+even bond pair);
   seed-cluster bootstrap over (L, seed).
Output: reentrant-exit-anatomy-v1/analysis/ed_v2_weaklink_controls.json
"""

import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "random_ising_gamma_ensemble"
OUT = BASE / "reentrant-exit-anatomy-v1" / "analysis"
N_BOOT = 2000


def load_cases():
    rows = []
    for folder in sorted(glob.glob(str(BASE / "pbc-reentrant-H1-ED-bulk-matched-L*-turing-v2"))):
        for path in sorted(glob.glob(folder + "/cases/L*/*.npz")):
            meta = json.loads(Path(path[:-4] + ".json").read_text(encoding="utf-8"))
            z = np.load(path)
            L, J, G, C = int(meta["L"]), z["J"], z["Gamma"], z["Cstag"]
            start = int(meta["H1_best_start_0based"])
            sites = [(start + t) % L for t in range(6)]
            gaps = []
            for a in range(L):
                i0, i1, i2, i3 = [(a + t) % L for t in range(4)]
                gaps.append(min(C[i0, i1], C[i0, i2], C[i1, i3], C[i2, i3]) - max(C[i1, i2], C[i0, i3]))
            gaps = np.array(gaps)
            rows.append({"L": L, "delta": float(meta["delta_Gamma0"]), "seed": int(meta["seed"]),
                         "log_gap": float(np.log(meta["gap01"])), "H1": float(meta["H1_max_lifetime_w6"]) > 1e-8,
                         "bulk": float(meta["bulk_ms2"]), "Jwin_mean": J[sites].mean(), "Gwin_mean": G[sites].mean(),
                         "sin": np.sin(2 * np.pi * start / L), "cos": np.cos(2 * np.pi * start / L),
                         "Jwin_min": J[sites].min(), "Jmin": J.min(), "pair": J[1::2].min() + J[0::2].min(),
                         "motif_J": np.array([J[(a + 1) % L] for a in range(L)])[gaps > 1e-8], "allJ": J})
    return rows


def main():
    rows = load_cases()
    y = np.array([r["log_gap"] for r in rows])
    H = np.array([r["H1"] for r in rows], float)
    strata = [(r["L"], r["delta"]) for r in rows]
    dummies = [np.array([s == lv for s in strata], float) for lv in sorted(set(strata))]
    clusters = np.array([(r["L"], r["seed"]) for r in rows])
    unique = np.unique(clusters, axis=0)
    members = {tuple(u): np.flatnonzero((clusters[:, 0] == u[0]) & (clusters[:, 1] == u[1])) for u in unique}
    rng = np.random.default_rng(20261013)
    draws = [np.concatenate([members[tuple(u)] for u in unique[rng.integers(0, len(unique), len(unique))]])
             for _ in range(N_BOOT)]

    def h1_coef(keys):
        X = np.column_stack([H] + [np.array([r[k] for r in rows], float) for k in keys] + dummies)
        est = np.linalg.lstsq(X, y, rcond=None)[0][0]
        boot = [np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][0] for idx in draws]
        return {"coefficient": float(est), "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])]}

    base = ["bulk", "Jwin_mean", "Gwin_mean", "sin", "cos"]
    motif = np.concatenate([r["motif_J"] for r in rows])
    allJ = np.concatenate([r["allJ"] for r in rows])
    summary = {
        "cases": len(rows), "cohort": "v2 only (seeds 1300-1349); v3 not available locally",
        "motif_location": {"n_motifs": int(motif.size), "middle_bond_J_median": float(np.median(motif)),
                           "all_bonds_J_median": float(np.median(allJ)), "mann_whitney_p": float(mannwhitneyu(motif, allJ).pvalue)},
        "H1_coefficient_on_log_gap": {
            "frozen_controls": h1_coef(base),
            "plus_window_min_J": h1_coef(base + ["Jwin_min"]),
            "plus_chain_min_J": h1_coef(base + ["Jmin"]),
            "plus_weakest_odd_even_pair": h1_coef(base + ["pair"]),
            "plus_all_weak_link_terms": h1_coef(base + ["Jwin_min", "Jmin", "pair"]),
        },
    }
    (OUT / "ed_v2_weaklink_controls.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
