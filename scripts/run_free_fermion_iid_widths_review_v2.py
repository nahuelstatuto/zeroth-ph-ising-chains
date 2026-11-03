"""Referee 1, round 2 (3.4): widths of the pseudocritical points with the unshifted i.i.d. ensemble.

Same disorder draws as the paper (generate_disorder_base(L, seed), seeds 0-499), same grid and
same estimators, but with ensemble="raw": J_i = 1 + u_i and Gamma_i = Gamma0 (1 + v_i) without the
shift to fixed sample means. The fixed-mean ensemble removes part of the sample-to-sample
fluctuations of delta = <ln Gamma> - <ln J>, which set the IDFP width ~ L^(-1/2). Exact free
fermions at h = 0, OBC, C = Cstag raw. Estimators: xi_lambda [Eq. (6)] and the physical response
-d m^2_{s,B}/dGamma0, maxima with the paper's three-point rule (analyze_free_fermion_h0_review_v1.peaks).
The fixed-mean maxima of the same seeds are read from results/review_v1/free-fermion-h0-v1 for a
paired comparison.
Run:  python scripts/run_free_fermion_iid_widths_review_v2.py
Output: results/review_v2/iid-widths-v1/{peaks_L*.npz, summary.json}
"""

import json
import os
import sys
import time
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
from analyze_free_fermion_h0_review_v1 import closed_form_xi, peaks, power  # noqa: E402

OUT = ROOT / "results" / "review_v2" / "iid-widths-v1"
FF = ROOT / "results" / "review_v1" / "free-fermion-h0-v1"
GRID = np.round(np.arange(0.5, 1.5 + 1e-9, 0.005), 6)
SIZES = (16, 32, 64, 128)
N = 500
WORKERS = 12
N_BOOT = 1000
RNG = np.random.default_rng(20261020)


def task(args):
    L, seed = args
    base = generate_disorder_base(L, seed)
    deaths = np.zeros((GRID.size, L - 1))
    m2 = np.zeros(GRID.size)
    B = slice(L // 4, L // 4 + L // 2)
    for k, g in enumerate(GRID):
        r = base.realize(float(g), ensemble="raw")
        C = cstag_elimination(majorana_state(r.J, r.Gamma)[0])
        D = np.clip(1.0 - C, 0.0, 2.0) + 1.0
        np.fill_diagonal(D, 0.0)
        deaths[k] = np.sort(minimum_spanning_tree(np.triu(D)).tocoo().data - 1.0)
        m2[k] = C[B, B].mean()
    xi = closed_form_xi(deaths, GRID)
    phys = -np.gradient(m2, GRID, edge_order=2)
    (px, ex), (pp, ep) = peaks(GRID, xi[None]), peaks(GRID, phys[None])
    return seed, float(px[0]), bool(ex[0]), float(pp[0]), bool(ep[0])


def fixed_mean_peaks(L):
    from analyze_free_fermion_h0_review_v1 import curves
    z = np.load(FF / f"random_L{L}.npz")
    g, _, cv = curves(z)
    seeds = z["seeds"].astype(int)
    keep = seeds < N
    out = {}
    for e in ("xi", "phys"):
        p, edge = peaks(g, cv[e][keep])
        out[e] = (p, edge)
    return seeds[keep], out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res, samples = {}, {"raw": {}, "fixed_mean": {}}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for L in SIZES:
            t0 = time.time()
            rows = sorted(pool.map(task, [(L, s) for s in range(N)], chunksize=4))
            seeds = np.array([r[0] for r in rows])
            arr = {"xi": (np.array([r[1] for r in rows]), np.array([r[2] for r in rows])),
                   "phys": (np.array([r[3] for r in rows]), np.array([r[4] for r in rows]))}
            np.savez_compressed(OUT / f"peaks_L{L}.npz", seeds=seeds, Gamma0=GRID,
                                xi=arr["xi"][0], xi_edge=arr["xi"][1], phys=arr["phys"][0], phys_edge=arr["phys"][1])
            fseeds, fm = fixed_mean_peaks(L)
            if not np.array_equal(fseeds, seeds):
                raise ValueError("seed mismatch with the fixed-mean data")
            res[f"L{L}"] = {}
            for e in ("xi", "phys"):
                ok = ~arr[e][1] & ~fm[e][1]
                samples["raw"].setdefault(e, {})[L] = arr[e][0][ok]
                samples["fixed_mean"].setdefault(e, {})[L] = fm[e][0][ok]
                res[f"L{L}"][e] = {"n": int(ok.sum()), "edge_raw": int(arr[e][1].sum()),
                                   "width_raw": float(arr[e][0][ok].std(ddof=1)), "width_fixed_mean": float(fm[e][0][ok].std(ddof=1)),
                                   "mean_raw": float(arr[e][0][ok].mean()), "mean_fixed_mean": float(fm[e][0][ok].mean())}
            print(f"L{L} {time.time() - t0:.0f} s  " + "  ".join(
                f"{e}: raw {res[f'L{L}'][e]['width_raw']:.4f} fm {res[f'L{L}'][e]['width_fixed_mean']:.4f}" for e in ("xi", "phys")), flush=True)
    fits = {}
    sizes = np.array(SIZES)
    for ens in ("raw", "fixed_mean"):
        for e in ("xi", "phys"):
            s = samples[ens][e]
            pw = power(sizes, np.array([s[L].std(ddof=1) for L in SIZES]))[0]
            bw = [power(sizes, np.array([s[L][RNG.integers(0, s[L].size, s[L].size)].std(ddof=1) for L in SIZES]))[0]
                  for _ in range(N_BOOT)]
            loc = float(np.log(s[SIZES[-2]].std(ddof=1) / s[SIZES[-1]].std(ddof=1)) / np.log(SIZES[-1] / SIZES[-2]))
            fits[f"{ens}_{e}"] = {"p_width": pw, "p_width_ci95": np.percentile(bw, [2.5, 97.5]).tolist(), "local_exponent_last": loc}
            print(ens, e, json.dumps(fits[f"{ens}_{e}"]), flush=True)
    (OUT / "summary.json").write_text(json.dumps({"sizes": list(SIZES), "realizations": N, "per_size": res, "fits": fits}, indent=2),
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
