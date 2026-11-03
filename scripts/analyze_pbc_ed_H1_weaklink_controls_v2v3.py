"""Weak-link controls on the frozen v2+v3 PBC ED H1 / low-gap association.

Reproduces the frozen controlled regression of
scripts/analyze_pbc_reentrant_H1_replication_v3.py (stratum fixed effects for
cohort x L x deltaGamma, H1 active, H1 x size, H1 x cohort, and the five
standardized controls; seed bootstrap within each cohort-size stratum), then
adds standardized weak-link covariates:
  window_min_J  : minimum J inside the selected H1 window (6 sites);
  chain_min_J   : minimum J of the ring;
  weakest_pair  : min J on odd bonds + min J on even bonds.
Also reports where the four-site H1 motifs sit (middle-bond J vs all bonds).

The ED cases are read from the archive given as first argument (default: the
user's Desktop copy); v2 cases fall back to results/ if absent there.
Output: results/.../reentrant-exit-anatomy-v1/analysis/ed_v2v3_weaklink_controls.json
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "random_ising_gamma_ensemble" / "reentrant-exit-anatomy-v1" / "analysis"
DEFAULT_ARCHIVE = ROOT / "results" / "results_random_ed_20260926"
SIZES = (12, 14, 16, 18)
DELTAS = (-0.025, 0.0, 0.025)
COHORTS = {"v2": tuple(range(1300, 1350)), "v3": tuple(range(1350, 1400))}
N_BOOT = 5000


def case_files(archive):
    v2 = sorted(glob.glob(str(archive / "pbc-reentrant-H1-ED-bulk-matched-L*-turing-v2" / "cases" / "L*" / "*.json")))
    if not v2:
        v2 = sorted(glob.glob(str(ROOT / "results" / "random_ising_gamma_ensemble" / "pbc-reentrant-H1-ED-bulk-matched-L*-turing-v2" / "cases" / "L*" / "*.json")))
    v3 = sorted(glob.glob(str(archive / "pbc-reentrant-H1-ED-bulk-matched-s1350-1399-worker1-v3" / "cases" / "*" / "*.json")))
    return v2, v3


def load(archive):
    v2, v3 = case_files(archive)
    data = {}
    motif_J, all_J = [], []
    for cohort, files in (("v2", v2), ("v3", v3)):
        for path in files:
            meta = json.loads(Path(path).read_text(encoding="utf-8"))
            with np.load(path[:-5] + ".npz", allow_pickle=False) as z:
                J, G, C = np.asarray(z["J"], float), np.asarray(z["Gamma"], float), np.asarray(z["Cstag"], float)
            L, seed, delta = int(meta["L"]), int(meta["seed"]), round(float(meta["delta_Gamma0"]), 12)
            start = int(meta["H1_best_start_0based"])
            sites = [(start + t) % L for t in range(6)]
            for a in range(L):
                i0, i1, i2, i3 = [(a + t) % L for t in range(4)]
                if min(C[i0, i1], C[i0, i2], C[i1, i3], C[i2, i3]) - max(C[i1, i2], C[i0, i3]) > 1e-8:
                    motif_J.append(J[(a + 1) % L])
            all_J.extend(J)
            data[(cohort, L, seed, delta)] = {
                "gap01": float(meta["gap01"]), "H1_active": float(float(meta["H1_max_lifetime_w6"]) > 1e-8),
                "bulk_ms2": float(meta["bulk_ms2"]), "J_mean": J[sites].mean(), "G_mean": G[sites].mean(),
                "sin": np.sin(2 * np.pi * start / L), "cos": np.cos(2 * np.pi * start / L),
                "window_min_J": J[sites].min(), "chain_min_J": J.min(), "weakest_pair": J[1::2].min() + J[0::2].min()}
    expected = {(c, L, s, round(d, 12)) for c, seeds in COHORTS.items() for L in SIZES for s in seeds for d in DELTAS}
    if set(data) != expected:
        raise RuntimeError(f"case set mismatch: missing {len(expected - set(data))}, extra {len(set(data) - expected)}")
    return data, np.array(motif_J), np.array(all_J)


def arrays(data):
    keys = [(c, L, s, round(d, 12)) for c in COHORTS for L in SIZES for s in COHORTS[c] for d in DELTAS]
    fields = list(next(iter(data.values())))
    return {f: np.array([data[k][f] for k in keys]) for f in fields}


def designs(A, extra):
    n = A["gap01"].size                                   # order: cohort(2) x size(4) x seed(50) x delta(3)
    stratum = np.repeat(np.arange(8), 150) * 3 + np.tile(np.arange(3), 400)
    cohort = np.repeat([0, 1], 600).astype(float)
    size = np.tile(np.repeat(np.asarray(SIZES, float), 150), 2)
    active = A["H1_active"]
    fixed = np.column_stack([(stratum == v).astype(float) for v in range(1, 24)])
    size_z = (size - size.mean()) / size.std(ddof=1)
    base = np.column_stack([np.ones(n), fixed, active, active * size_z, active * cohort])
    names = ["bulk_ms2", "J_mean", "G_mean", "sin", "cos"] + list(extra)
    cont = np.column_stack([A[k] for k in names])
    cont = (cont - cont.mean(0)) / cont.std(0, ddof=1)
    return np.column_stack([base, cont]), np.log(A["gap01"])


def h1_coefficient(X, y, rows=None):
    if rows is not None:
        X, y = X[rows], y[rows]
    return np.linalg.lstsq(X, y, rcond=None)[0][24]


def main():
    archive = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ARCHIVE
    data, motif_J, all_J = load(archive)
    A = arrays(data)
    rng = np.random.default_rng(2026091604)
    draws = []
    for _ in range(N_BOOT):
        rows = []
        for stratum in range(8):
            for seed in rng.integers(0, 50, 50):
                start = stratum * 150 + int(seed) * 3
                rows.extend((start, start + 1, start + 2))
        draws.append(np.asarray(rows))
    results = {}
    for name, extra in (("frozen_controls", ()), ("plus_window_min_J", ("window_min_J",)),
                        ("plus_chain_min_J", ("chain_min_J",)), ("plus_weakest_pair", ("weakest_pair",)),
                        ("plus_all_weak_link", ("window_min_J", "chain_min_J", "weakest_pair"))):
        X, y = designs(A, extra)
        est = h1_coefficient(X, y)
        boot = np.array([h1_coefficient(X, y, r) for r in draws])
        results[name] = {"H1_coefficient": float(est), "ci95": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
                         "fraction_boot_below_zero": float(np.mean(boot < 0))}
        print(f"{name:22s} {est:+.4f}  95% [{results[name]['ci95'][0]:+.4f}, {results[name]['ci95'][1]:+.4f}]", flush=True)
    summary = {
        "archive": str(archive), "cases": int(A["gap01"].size), "bootstrap": N_BOOT,
        "frozen_reference": {"H1_coefficient": -0.2652, "ci95": [-0.5375, -0.0089]},
        "H1_coefficient_on_log_gap": results,
        "motif_location": {"n_motifs": int(motif_J.size), "middle_bond_J_median": float(np.median(motif_J)),
                           "all_bonds_J_median": float(np.median(all_J)), "mann_whitney_p": float(mannwhitneyu(motif_J, all_J).pvalue)},
    }
    (OUT / "ed_v2v3_weaklink_controls.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["motif_location"], indent=1))


if __name__ == "__main__":
    main()
