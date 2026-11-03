"""B4 (review v3, R1-3): do the per-realization pseudocritical points follow each sample?

For every realization of the exact h = 0 campaign (results/review_v1/free-fermion-h0-v1, OBC,
fixed_mean, 1000 realizations for L = 16-128, 500 for L = 256) compare the maxima of xi_lambda
(topological) and of -d m2_bulk / dGamma0 (physical), taken from the hashed input of Fig. 3, with
two sample-specific references that need no ground state:
  * self-dual point: Gamma_i = Gamma0 g_i in fixed_mean, so <ln Gamma>_alpha = <ln J>_alpha gives
    ln Gamma0_sd = <ln J> (L-1 bonds) - <ln g> (L fields);
  * surface magnetization (Peschel 1984), m_s = [1 + sum_{l=1}^{L-1} prod_{j<=l} (Gamma_j/J_j)^2]^(-1/2),
    for the left end and, mirrored, for the right end. Its pseudocritical point is the maximum of
    -d m_s / dGamma0 on a fine grid; we use the mean of the two ends.
Reports Pearson and Spearman correlations with bootstrap intervals, the fraction of the variance of
Gamma*_alpha explained by each reference, and the same for the physical estimator as a benchmark.
Also: (i) self-dual points of the central half B (the window of m2_bulk) and of the two end
quarters, which the fixed arithmetic means anticorrelate; (ii) cross-validated R^2 of a linear
model of each estimator on the segment self-dual logarithms (K = 4, 8 segments); (iii) the
per-realization Pearson matrix of all benchmark estimators (analyze_free_fermion_h0_review_v1).
Output: results/review_v3/per-realization-pseudocritical-v1/summary.json (+ per_realization.npz).
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from clean_tfim_tn.disorder.reentrant import generate_disorder_base  # noqa: E402
import analyze_free_fermion_h0_review_v1 as an  # noqa: E402

ESTIMATORS = ("xi", "phys", "chiF", "ds", "d2e", "fro", "dcnn", "xi_nn")

PEAKS = ROOT / "paper_figures/fig_random_h0/data_raw/random_h0_peaks.npz"
OUT = ROOT / "results/review_v3/per-realization-pseudocritical-v1"
GRID = np.linspace(0.3, 2.0, 17001)
N_BOOT = 2000
RNG = np.random.default_rng(20261102)


def references(L, seed):
    base = generate_disorder_base(L, int(seed))
    r = base.realize(1.0, "fixed_mean")
    J, g = np.asarray(r.J), np.asarray(r.Gamma)  # Gamma0 = 1, so Gamma = g
    sd = float(np.exp(np.mean(np.log(J)) - np.mean(np.log(g))))
    out = [sd]
    for JJ, gg in ((J, g[:-1]), (J[::-1], g[::-1][:-1])):  # left end, right end
        a = np.cumsum(2 * np.log(gg / JJ))  # ln prod_{j<=l} (g_j/J_j)^2, l = 1..L-1
        l = np.arange(1, L)
        lnG = np.log(GRID)[:, None]
        e = a[None, :] + 2 * l[None, :] * lnG
        mx = np.maximum(e.max(axis=1), 0.0)
        s = np.exp(-mx) + np.exp(e - mx[:, None]).sum(axis=1)  # (1 + sum) * exp(-mx)
        ms = np.exp(-0.5 * (np.log(s) + mx))
        d = -np.gradient(ms, GRID)
        i = int(np.argmax(d))
        out.append(float(GRID[i]))
    return out


def segment_logs(L, seed, K):
    r = generate_disorder_base(L, int(seed)).realize(1.0, "fixed_mean")
    J, g = np.log(r.J), np.log(r.Gamma)
    e = np.linspace(0, L, K + 1).astype(int)
    return [J[e[k]:min(e[k + 1], L - 1)].mean() - g[e[k]:e[k + 1]].mean() for k in range(K)]


def central_and_ends(L, seed):
    r = generate_disorder_base(L, int(seed)).realize(1.0, "fixed_mean")
    J, g = np.log(r.J), np.log(r.Gamma)
    a, b = L // 4, 3 * L // 4   # B = sites a+1..b (1-based), as m2_bulk
    sd_b = np.exp(J[a:b - 1].mean() - g[a:b].mean())
    sd_e = np.exp(np.r_[J[:a], J[b - 1:]].mean() - np.r_[g[:a], g[b:]].mean())
    return float(sd_b), float(sd_e)


def cv_r2(X, y, folds=5):
    X = np.c_[np.ones(len(y)), X]
    idx = np.arange(len(y))
    sse = 0.0
    for k in range(folds):
        te = idx % folds == k
        coef = np.linalg.lstsq(X[~te], y[~te], rcond=None)[0]
        sse += float(((y[te] - X[te] @ coef) ** 2).sum())
    return 1.0 - sse / float(((y - y.mean()) ** 2).sum())


def all_peaks(L):
    z = np.load(an.DATA / f"random_L{L}.npz")
    fpath = an.DATA / f"fidelity_random_L{L}.npz"
    g, mid, cv = an.curves(z, np.load(fpath) if fpath.exists() else None)
    out = {}
    for key in ESTIMATORS:
        x = mid if key == "chiF" else g
        p, edge = an.peaks(x, cv[key])
        out[key] = np.where(edge, np.nan, p)
    return np.asarray(z["seeds"]), out


def stats(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    pr, sr = pearsonr(x, y)[0], spearmanr(x, y)[0]
    bp = []
    for _ in range(N_BOOT):
        k = RNG.integers(0, x.size, x.size)
        bp.append(pearsonr(x[k], y[k])[0])
    return {"n": int(x.size), "pearson": float(pr), "pearson_ci95": [float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))],
            "spearman": float(sr), "r2": float(pr ** 2)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = np.load(PEAKS)
    summary, arrays = {}, {}
    for L in [int(v) for v in P["sizes"]]:
        seeds = P[f"L{L}_seeds"]
        ref = np.array([references(L, s) for s in seeds])
        sd, ms_left, ms_right = ref.T
        ms = 0.5 * (ms_left + ms_right)
        top, phys = P[f"L{L}_xi"], P[f"L{L}_phys"]
        arrays.update({f"L{L}_seeds": seeds, f"L{L}_sd": sd, f"L{L}_ms_left": ms_left, f"L{L}_ms_right": ms_right,
                       f"L{L}_top": top, f"L{L}_phys": phys})
        edge = (ms_left <= GRID[1]) | (ms_left >= GRID[-2]) | (ms_right <= GRID[1]) | (ms_right >= GRID[-2])
        summary[f"L{L}"] = {
            "n_real": int(seeds.size), "ms_edge": int(edge.sum()),
            "mean": {"top": float(np.nanmean(top)), "phys": float(np.nanmean(phys)), "sd": float(sd.mean()), "ms": float(ms.mean())},
            "std": {"top": float(np.nanstd(top, ddof=1)), "phys": float(np.nanstd(phys, ddof=1)), "sd": float(sd.std(ddof=1)), "ms": float(ms.std(ddof=1))},
            "top_vs_sd": stats(top, sd), "top_vs_ms": stats(top, ms), "phys_vs_sd": stats(phys, sd),
            "phys_vs_ms": stats(phys, ms), "top_vs_phys": stats(top, phys), "sd_vs_ms": stats(sd, ms),
            "ms_left_vs_right": stats(ms_left, ms_right),
        }
        cb = np.array([central_and_ends(L, s_) for s_ in seeds])
        arrays.update({f"L{L}_sd_central": cb[:, 0], f"L{L}_sd_ends": cb[:, 1]})
        s = summary[f"L{L}"]
        s["std"]["sd_central"] = float(cb[:, 0].std(ddof=1))
        s["top_vs_sd_central"] = stats(top, cb[:, 0])
        s["phys_vs_sd_central"] = stats(phys, cb[:, 0])
        s["sd_central_vs_ends"] = stats(cb[:, 0], cb[:, 1])
        s["ms_vs_sd_ends"] = stats(ms, cb[:, 1])
        s["cv_r2_segments"] = {}
        for K in (4, 8):
            X = np.array([segment_logs(L, s_, K) for s_ in seeds])
            s["cv_r2_segments"][str(K)] = {"top": cv_r2(X, top), "phys": cv_r2(X, phys)}
        aseeds, pk = all_peaks(L)
        assert np.array_equal(aseeds, seeds)
        assert np.allclose(pk["xi"], top, equal_nan=True) and np.allclose(pk["phys"], phys, equal_nan=True)
        M = {}
        for i, a_ in enumerate(ESTIMATORS):
            for b_ in ESTIMATORS[i + 1:]:
                ok = np.isfinite(pk[a_]) & np.isfinite(pk[b_])
                M[f"{a_}|{b_}"] = float(pearsonr(pk[a_][ok], pk[b_][ok])[0])
        s["estimator_pearson"] = M
        print(L, {k: round(s[k]["pearson"], 3) for k in s if isinstance(s[k], dict) and "pearson" in s[k]})
    np.savez_compressed(OUT / "per_realization.npz", **arrays)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
