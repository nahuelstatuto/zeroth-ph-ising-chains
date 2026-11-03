"""Referees 1 (3.1) and 2 (2.2): how much does the result depend on the staggered input?

Three dissimilarities built from the same ground states:
  stag   D = 1 - C,     C = (-1)^|i-j| <Z_i Z_j>   (paper)
  abs    D = 1 - |C|                              (sign-blind)
  raw    D = 1 - <Z_i Z_j>                        (unstaggered; at h = 0 also the connected one, <Z_i> = 0)
For each, xi_lambda from Eq. (6) on the minimum-spanning-tree deaths and its maximum (quadratic
refinement). At h = 0 the staggered correlations are all positive (Griffiths), so abs = stag
there and only raw is tested.
  Part A, h = 0, exact free fermions: clean L = 16, 32, 64, 128 and random L = 16, 32, 64
          (300 realizations, Gamma0 = 0.5-1.5, step 0.01). Peaks and distribution widths.
  Part B, h > 0, DMRG data of the odd chains L = 31 (Fig. 5 cuts, QC mask of the figure):
          mean absolute deviation of the maxima from the reference boundary.
Run:  python scripts/test_correlator_choice_review_v1.py
Output: results/review_v1/correlator-choice-v1/summary.json
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from clean_tfim_tn.disorder.reentrant import generate_disorder_base  # noqa: E402
from clean_tfim_tn.exact.free_fermion_h0 import cstag_elimination, majorana_state  # noqa: E402

OUT = ROOT / "results" / "review_v1" / "correlator-choice-v1"
GRID = np.round(np.arange(0.5, 1.5 + 1e-9, 0.01), 4)
KINDS = ("stag", "abs", "raw")


def deaths(D):
    M = np.clip(D, 0.0, 2.0) + 1.0
    np.fill_diagonal(M, 0.0)
    return np.sort(minimum_spanning_tree(np.triu(M)).tocoo().data - 1.0)


def dissimilarities(C):
    L = C.shape[0]
    sign = (-1.0) ** np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    return {"stag": 1.0 - C, "abs": 1.0 - np.abs(C), "raw": 1.0 - sign * C}


def xi(g, d):
    dd = np.gradient(d, g, axis=0, edge_order=2)
    return np.sqrt((d * dd ** 2).sum(1) / (2 * d.shape[1]))


def peak(g, y, allowed=None):
    yy = np.where(allowed, y, -np.inf) if allowed is not None else y
    i = int(np.argmax(yy[1:-1])) + 1
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    return float(g[i] + (0.5 * (a - c) / den * (g[1] - g[0]) if den < 0 else 0.0))


def scan_ff(args):
    L, seed = args
    if seed is None:
        J_of = lambda g: (np.ones(L - 1), np.full(L, float(g)))  # noqa: E731
    else:
        base = generate_disorder_base(L, seed)
        J_of = lambda g: (base.realize(float(g)).J, base.realize(float(g)).Gamma)  # noqa: E731
    ds = {k: [] for k in KINDS}
    for g in GRID:
        C = cstag_elimination(majorana_state(*J_of(g))[0])
        for k, D in dissimilarities(C).items():
            ds[k].append(deaths(D))
    return L, seed, {k: peak(GRID, xi(GRID, np.array(v))) for k, v in ds.items()}


def part_b():
    import benchmark_estimators_clean_frontier_review_v1 as bf
    import analyze_clean_frontier_multisize as frontier
    z = np.load(bf.FIG / "data_raw" / "clean_frontier_oddL_compact.npz")
    best = {}
    for e in json.loads(str(z["index_json"])):
        key = (e["L"], round(e["h"], 4))
        if e["L"] == 31 and round(e["h"], 4) in bf.GRID_H and (key not in best or e["chi"] > best[key]["chi"]):
            best[key] = e
    err = {k: [] for k in KINDS}
    for (L, h), e in sorted(best.items()):
        scans = []
        for s in e["sources"]:
            p = Path(s) if Path(s).is_absolute() else ROOT / s
            with np.load(p) as a:
                scans.append({"gamma": np.asarray(a["Gamma_values"], float), "cstag": np.asarray(a["Cstag"], float),
                              "energy": np.asarray(a["selected_energy_density"], float),
                              "entropy": np.asarray(a["half_chain_entropy"], float),
                              "bond": np.asarray(a["selected_max_bond_dimension"], int),
                              "branch": np.asarray(a["branch_disagreement"], bool),
                              "energy_diff": np.asarray(a["absolute_energy_difference"], float),
                              "fidelity": np.asarray(a["branch_fidelity"], float)})
        d = frontier.merge(scans)
        g, C = d["gamma"], np.asarray(d["cstag"], float)
        flag = np.asarray(d["branch"], bool) | (np.asarray(d["bond"]) >= e["chi"])
        allowed = ~(flag | np.r_[False, flag[:-1]] | np.r_[flag[1:], False])
        ref = float(np.interp(h, bf.REF_H, bf.REF_G))
        per = {k: [] for k in KINDS}
        for m in C:
            for k, D in dissimilarities(m).items():
                per[k].append(deaths(D))
        for k in KINDS:
            err[k].append(abs(bf.masked_peak(g, bf.velocity(g, np.array(per[k]), L), allowed)["x"] - ref))
    return {k: {"MAE": float(np.mean(v)), "max": float(np.max(v)), "cuts": len(v)} for k, v in err.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(L, None) for L in (16, 32, 64, 128)] + [(L, s) for L in (16, 32, 64) for s in range(300)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        res = list(pool.map(scan_ff, jobs, chunksize=4))
    summary = {"clean_h0": {}, "random_h0": {}}
    for L, seed, pk in res:
        if seed is None:
            summary["clean_h0"][f"L{L}"] = pk
    for L in (16, 32, 64):
        P = {k: np.array([pk[k] for LL, s, pk in res if LL == L and s is not None]) for k in KINDS}
        summary["random_h0"][f"L{L}"] = {
            **{f"{k}_mean": float(P[k].mean()) for k in KINDS}, **{f"{k}_width": float(P[k].std(ddof=1)) for k in KINDS},
            "median_abs_diff_raw_vs_stag": float(np.median(np.abs(P["raw"] - P["stag"]))),
            "max_abs_diff_abs_vs_stag": float(np.max(np.abs(P["abs"] - P["stag"])))}
    print(json.dumps(summary, indent=1), flush=True)
    summary["clean_frontier_L31"] = part_b()
    print(json.dumps(summary["clean_frontier_L31"], indent=1), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
