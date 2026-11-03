"""B9 (review v3, R1 minor): the unweighted merge-scale velocity against xi_lambda.

R1: "the weighting d_k of Eq. (6) is fixed by the landscape, but is it physically motivated? Show in
the SM the unweighted version, sum (d d_k)^2, which is the velocity of the edges of the minimum
spanning tree."

    xi_lambda^2 = (1/2) sum_k d_(k) (dg d_(k))^2        (Eq. (6), landscape velocity)
    v_0^2       =       sum_k       (dg d_(k))^2        (unweighted, speed of the sorted merge scales)

Same data, derivatives and peak rules as the paper for the three uses of xi_lambda:
  1. clean chain, h = 0 (Sec. III, Fig. 2): exact free fermions; the MST is the chain, so the merge
     scales are the nearest-neighbour bars 1 - C_{i,i+1}. Maxima refined as in B5
     (analyze_clean_fss_systematics_review_v3: local grid of step 1e-4, three-point parabola),
     fits of g*(L) with the ansatzes of B5.
  2. random chain, h = 0 (Sec. IV A, Fig. 3): sorted deaths of results/review_v1/free-fermion-h0-v1
     (1000 realizations for L = 16-128, 500 for L = 256), three-point rule of
     analyze_free_fermion_h0_review_v1.peaks; mean, width, width exponent and per-realization
     correlation with the xi_lambda and the physical maxima.
  3. clean frontier, h > 0 (Sec. V, Table II): cuts, merge, QC mask and peak rule of
     benchmark_estimators_clean_frontier_review_v1, with its landscape integral replaced by v_0.
OBC, raw C = Cstag, D = 1 - C.
Run (tn_tda):  python scripts/analyze_unweighted_velocity_review_v3.py
Output: results/review_v3/unweighted-velocity-v1/summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_clean_fss_systematics_review_v3 as fss  # noqa: E402
import analyze_free_fermion_h0_review_v1 as ffa  # noqa: E402
import benchmark_estimators_clean_frontier_review_v1 as bf  # noqa: E402

OUT = ROOT / "results" / "review_v3" / "unweighted-velocity-v1"
DATA = ROOT / "results" / "review_v1" / "free-fermion-h0-v1"
FSS_SUMMARY = ROOT / "results" / "review_v3" / "clean-fss-systematics-v1" / "summary.json"
N_BOOT = 1000
RNG = np.random.default_rng(20261102)


def v0(bars, g):
    """Unweighted velocity along the field axis (-2), bars sorted along the last axis."""
    d = np.gradient(bars, g, axis=-2, edge_order=2)
    return np.sqrt((d ** 2).sum(-1) / bars.shape[-1])


def clean_curves(L, grid):
    bars = np.array([np.sort(1.0 - (-np.diagonal(fss.W_of(L, g), -1))) for g in grid])
    return ffa.closed_form_xi(bars, grid), v0(bars, grid)


def clean_part():
    out, gx, gv = {}, [], []
    for L in fss.SIZES:
        z = np.load(DATA / (f"clean_L{L}_full.npz" if L == 512 else f"clean_L{L}.npz"))
        g = np.asarray(z["Gamma0"], float)
        deaths = z["deaths"].astype(float)
        deaths = deaths.reshape(-1, g.size, deaths.shape[-1])[0] if deaths.ndim == 3 else deaths
        coarse = {"xi": ffa.closed_form_xi(deaths, g), "v0": v0(deaths, g)}
        row = {}
        for name, k in (("xi", 0), ("v0", 1)):
            g0 = g[int(np.argmax(coarse[name]))]
            grid = np.arange(g0 - fss.HALF, g0 + fss.HALF + fss.STEP / 2, fss.STEP)
            row[name] = fss.parabola_peak(grid, clean_curves(L, grid)[k])
            row[f"{name}_coarse"] = float(ffa.peaks(g, coarse[name])[0][0])
        out[f"L{L}"] = row
        gx.append(row["xi"])
        gv.append(row["v0"])
        print(f"clean L{L}: g* xi {row['xi']:.5f}  v0 {row['v0']:.5f}", flush=True)
    Ls = np.array(fss.SIZES, float)
    out["fits_xi"] = fss.fits(Ls, np.array(gx))
    out["fits_v0"] = fss.fits(Ls, np.array(gv))
    ref = json.loads(FSS_SUMMARY.read_text(encoding="utf-8"))["peaks"]["xi"]
    out["check_xi_vs_B5_max_abs_dev"] = float(max(abs(out[f"L{L}"]["xi"] - ref[str(L)]) for L in fss.SIZES))
    return out


def random_part():
    out, samples = {}, {"xi": {}, "v0": {}, "phys": {}}
    for L in (16, 32, 64, 128, 256):
        z = np.load(DATA / f"random_L{L}.npz")
        g = np.asarray(z["Gamma0"], float)
        deaths = z["deaths"].astype(float)
        cv = {"xi": ffa.closed_form_xi(deaths, g), "v0": v0(deaths, g),
              "phys": -np.gradient(z["m2_bulk"], g, axis=-1, edge_order=2)}
        pk = {k: ffa.peaks(g, v) for k, v in cv.items()}
        ok = ~(pk["xi"][1] | pk["v0"][1] | pk["phys"][1])
        row = {"n": int(deaths.shape[0]), "n_interior": int(ok.sum())}
        for k, (p, e) in pk.items():
            row[k] = {"mean": float(p[~e].mean()), "width": float(p[~e].std(ddof=1)), "edge_fraction": float(e.mean())}
            samples[k][L] = p[~e]
        row["pearson_v0_xi"] = float(pearsonr(pk["v0"][0][ok], pk["xi"][0][ok])[0])
        row["spearman_v0_xi"] = float(spearmanr(pk["v0"][0][ok], pk["xi"][0][ok])[0])
        row["pearson_v0_phys"] = float(pearsonr(pk["v0"][0][ok], pk["phys"][0][ok])[0])
        row["pearson_xi_phys"] = float(pearsonr(pk["xi"][0][ok], pk["phys"][0][ok])[0])
        row["median_abs_diff_v0_xi"] = float(np.median(np.abs(pk["v0"][0][ok] - pk["xi"][0][ok])))
        out[f"L{L}"] = row
        print(f"random L{L}: mean/width xi {row['xi']['mean']:.4f}/{row['xi']['width']:.4f}  "
              f"v0 {row['v0']['mean']:.4f}/{row['v0']['width']:.4f}  r(v0,xi) {row['pearson_v0_xi']:.3f}  "
              f"r(v0,phys) {row['pearson_v0_phys']:.3f}  r(xi,phys) {row['pearson_xi_phys']:.3f}", flush=True)
    for lmin in (16, 32):
        for k in samples:
            use = np.array([L for L in samples[k] if L >= lmin])
            w = np.array([samples[k][L].std(ddof=1) for L in use])
            bw = [ffa.power(use, np.array([s[RNG.integers(0, s.size, s.size)].std(ddof=1)
                                           for s in (samples[k][L] for L in use)]))[0] for _ in range(N_BOOT)]
            out[f"p_width_Lmin{lmin}_{k}"] = {"p": ffa.power(use, w)[0], "ci95": np.percentile(bw, [2.5, 97.5]).tolist()}
    return out


def frontier_part():
    weighted = bf.velocity

    def unweighted(g, bars, L):
        return v0(bars, g)

    res = {}
    for label, velocity in (("xi", weighted), ("v0", unweighted)):
        bf.velocity = velocity
        cuts = {**bf.load("clean_frontier_compact.npz", (16, 32, 64), odd=False),
                **bf.load("clean_frontier_oddL_compact.npz", (31, 63), odd=True)}
        res[label] = cuts
    bf.velocity = weighted
    out = {}
    for L in sorted({k[0] for k in res["xi"]}):
        row = {}
        for label in ("xi", "v0"):
            err = [abs(c["peaks"]["xi"]["x"] - c["ref"]) for k, c in res[label].items() if k[0] == L]
            row[label] = {"MAE": float(np.mean(err)), "max": float(np.max(err)), "n": len(err),
                          "edge": int(sum(c["peaks"]["xi"]["edge"] for k, c in res[label].items() if k[0] == L))}
        row["phys_MAE"] = float(np.mean([abs(c["peaks"]["phys"]["x"] - c["ref"]) for k, c in res["xi"].items() if k[0] == L]))
        row["max_abs_peak_diff"] = float(max(abs(res["v0"][k]["peaks"]["xi"]["x"] - res["xi"][k]["peaks"]["xi"]["x"])
                                             for k in res["xi"] if k[0] == L))
        out[f"L{L}"] = row
        print(f"frontier L{L}: MAE xi {row['xi']['MAE']:.4f}  v0 {row['v0']['MAE']:.4f}  phys {row['phys_MAE']:.4f}  "
              f"max |peak diff| {row['max_abs_peak_diff']:.4f}", flush=True)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"clean_h0": clean_part(), "random_h0": random_part(), "frontier": frontier_part()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
