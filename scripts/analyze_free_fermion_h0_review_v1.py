"""Referee benchmark at h = 0 with exact free fermions (review v1).

Input: results/review_v1/free-fermion-h0-v1/{clean,random}_L*.npz
(scripts/run_free_fermion_h0_review_v1.py). Estimators, each maximized over Gamma0
(interior maximum, quadratic refinement; maxima at the grid ends are counted as edge):

  phys   -d m^2_{s,B} / dGamma0                (paper physical response)
  xi     H0 landscape velocity, closed form Eq. (6): xi^2 = sum_k d_k (d_k')^2 / (2 (L - 1))
  xi_nn  same closed form on the sorted nearest-neighbour bars 1 - C_{i,i+1}
  dcnn   |d mean_i C_{i,i+1} / dGamma0|         (bond-energy response)
  fro    ||dC/dGamma0||_F / L
  d2e    -d^2 (E/L) / dGamma0^2
  ds     |d S_{L/2} / dGamma0|
  chiF   fidelity susceptibility -2 ln F / dGamma0^2 (midpoints)

Random chain: shift |mean - 1| and width std(ddof=1) of the per-realization maxima,
power laws on L >= 16 (paper) and L >= 32, realization bootstrap of the exponents,
KS collapse of (peak - mean) L^a (mean pairwise KS distance, paper definition) with a
bootstrap interval for the best exponent, edge fractions, MST locality (fraction of
nearest-neighbour merge edges near Gamma0 = 1).
Clean chain: maxima per L, fits g*(L) = g_c + a/L, g_c + a/L + b/L^2, g_c + a L^(-1/nu)
(L >= 32 and L >= 64), and peak height of xi against ln L.

Run:  python scripts/analyze_free_fermion_h0_review_v1.py
Output: results/review_v1/free-fermion-h0-v1/analysis/summary.json
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "review_v1" / "free-fermion-h0-v1"
OUT = DATA / "analysis"
EST = ("phys", "xi", "xi_nn", "dcnn", "fro", "d2e", "ds", "chiF")
EXPONENTS = np.round(np.arange(0.0, 1.201, 0.01), 3)
N_BOOT = 1000
RNG = np.random.default_rng(20261016)


def closed_form_xi(bars, g):
    """Eq. (6) along the last-but-one axis (field), bars sorted along the last axis."""
    d = np.gradient(bars, g, axis=-2, edge_order=2)
    return np.sqrt((bars * d ** 2).sum(-1) / (2 * bars.shape[-1]))


def curves(z, fid=None):
    g = np.asarray(z["Gamma0"], float)
    L = int(z["L"])
    deaths = z["deaths"].astype(float)
    cnn = z["cnn"].astype(float)
    nn_bars = np.sort(np.clip(1.0 - cnn, 0.0, 2.0), axis=-1)
    grad = lambda y: np.gradient(y, g, axis=-1, edge_order=2)  # noqa: E731
    out = {
        "phys": -grad(z["m2_bulk"]),
        "xi": closed_form_xi(deaths, g),
        "xi_nn": closed_form_xi(nn_bars, g),
        "dcnn": np.abs(grad(cnn.mean(-1))),
        "fro": z["dC_fro"],
        "d2e": -grad(grad(z["energy"] / L)),
        "ds": np.abs(grad(z["S_half"])),
    }
    mid = 0.5 * (g[1:] + g[:-1])
    logF = z["logF"] if fid is None else fid["logF"]       # parity-consistent fidelity when available
    out["chiF"] = -2.0 * logF / np.diff(g) ** 2
    return g, mid, out


def peaks(x, Y):
    """Interior maximum with quadratic refinement, per row; edge flag when at the ends or NaN-limited."""
    Y = np.atleast_2d(Y)
    Yf = np.where(np.isfinite(Y), Y, -np.inf)
    i = np.argmax(Yf, axis=1)
    edge = (i == 0) | (i == Y.shape[1] - 1)
    ii = np.clip(i, 1, Y.shape[1] - 2)
    r = np.arange(Y.shape[0])
    a, b, c = Yf[r, ii - 1], Yf[r, ii], Yf[r, ii + 1]
    edge |= ~np.isfinite(a) | ~np.isfinite(c)
    den = a - 2 * b + c
    dx = x[1] - x[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        off = np.where((den < 0) & ~edge, 0.5 * (a - c) / den * dx, 0.0)
    return x[ii] + off, edge


def power(Ls, y):
    m, c = np.polyfit(np.log(Ls), np.log(y), 1)
    return float(-m), float(np.exp(c))


def ks(a, b):
    x = np.sort(np.concatenate([a, b]))
    return float(np.max(np.abs(np.searchsorted(np.sort(a), x, side="right") / a.size
                               - np.searchsorted(np.sort(b), x, side="right") / b.size)))


def collapse(samples):
    mm = []
    for e in EXPONENTS:
        sc = {L: (s - s.mean()) * L ** e for L, s in samples.items()}
        mm.append(np.mean([ks(sc[a], sc[b]) for a, b in combinations(sorted(sc), 2)]))
    mm = np.array(mm)
    return float(EXPONENTS[int(np.argmin(mm))]), float(mm.min())


def random_part():
    res, raw = {}, {}
    files = sorted(DATA.glob("random_L*.npz"), key=lambda p: int(p.stem.split("L")[1]))
    for f in files:
        z = np.load(f)
        L = int(z["L"])
        ff = DATA / f"fidelity_{f.stem}.npz"
        g, mid, cv = curves(z, np.load(ff) if ff.exists() else None)
        pk = {}
        for e in EST:
            x = mid if e == "chiF" else g
            p, edge = peaks(x, cv[e])
            pk[e] = {"peaks": p, "edge": edge}
        near = np.abs(g - 1.0) <= 0.1
        nn_frac = float((z["sep"][:, near] == 1).mean())
        raw[L] = pk
        dm = ROOT / "paper_figures" / "fig_random_h0" / "data_raw" / "random_h0_fine50_scaling.npz"
        if dm.exists():
            zz = np.load(dm)
            if f"L{L}_seeds" in zz.files:
                pub = dict(zip(zz[f"L{L}_seeds"].astype(int), zz[f"L{L}_landscape_Gamma0_star_by_seed"]))
                mine = dict(zip(z["seeds"].astype(int), pk["xi"]["peaks"]))
                common = [s for s in pub if s in mine]
                diff = np.array([mine[s] - pub[s] for s in common])
                dmrg_check = {"seeds": len(common), "median_abs_diff": float(np.median(np.abs(diff))),
                              "max_abs_diff": float(np.max(np.abs(diff))),
                              "dmrg_grid_step": float(np.diff(zz[f"L{L}_Gamma0"]).min())}
            else:
                dmrg_check = None
        res[f"L{L}"] = {"realizations": int(z["deaths"].shape[0]), "grid": [float(g[0]), float(g[-1]), int(g.size)],
                        "mst_nn_fraction_near_critical": nn_frac, "dmrg_check": dmrg_check,
                        **{e: {"mean": float(pk[e]["peaks"][~pk[e]["edge"]].mean()),
                               "width": float(pk[e]["peaks"][~pk[e]["edge"]].std(ddof=1)),
                               "edge_fraction": float(pk[e]["edge"].mean())} for e in EST}}
        print(f"random L{L}: " + "  ".join(f"{e} {res[f'L{L}'][e]['mean']:.4f}/{res[f'L{L}'][e]['width']:.4f}" for e in EST)
              + f" | NN frac {nn_frac:.3f}", flush=True)
    sizes = np.array(sorted(raw))
    fits = {}
    for minL in (16, 32):
        use = sizes[sizes >= minL]
        for e in EST:
            samples = {L: raw[L][e]["peaks"][~raw[L][e]["edge"]] for L in use}
            shift = np.array([abs(samples[L].mean() - 1.0) for L in use])
            width = np.array([samples[L].std(ddof=1) for L in use])
            p_s, _ = power(use, shift)
            p_w, _ = power(use, width)
            bs, bw, bk = [], [], []
            for _ in range(N_BOOT):
                rs = {L: s[RNG.integers(0, s.size, s.size)] for L, s in samples.items()}
                bs.append(power(use, np.array([abs(rs[L].mean() - 1.0) for L in use]))[0])
                bw.append(power(use, np.array([rs[L].std(ddof=1) for L in use]))[0])
            for _ in range(200):
                rs = {L: s[RNG.integers(0, s.size, s.size)] for L, s in samples.items()}
                bk.append(collapse(rs)[0])
            a, mm = collapse(samples)
            fits[f"Lmin{minL}_{e}"] = {"sizes": use.tolist(), "p_shift": p_s, "p_shift_ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
                                       "p_width": p_w, "p_width_ci95": np.percentile(bw, [2.5, 97.5]).tolist(),
                                       "ks_exponent": a, "ks_exponent_ci95": np.percentile(bk, [2.5, 97.5]).tolist(),
                                       "ks_mismatch": mm}
            print(f"Lmin{minL} {e}: p_w {p_w:.3f} {np.round(np.percentile(bw, [2.5, 97.5]), 3)}  p_s {p_s:.3f}  KS a {a:.2f} {np.round(np.percentile(bk, [2.5, 97.5]), 2)}", flush=True)
    return res, fits


def clean_part():
    res, table = {}, {}
    for f in sorted(DATA.glob("clean_L*.npz"), key=lambda p: int(p.stem.split("L")[1])):
        z = np.load(f)
        L = int(z["L"])
        ff = DATA / f"fidelity_{f.stem}.npz"
        g, mid, cv = curves(z, np.load(ff) if ff.exists() else None)
        row = {}
        for e in EST:
            p, edge = peaks(mid if e == "chiF" else g, cv[e])
            row[e] = {"peak": float(p[0]), "edge": bool(edge[0]), "height": float(np.nanmax(cv[e][0]))}
        res[f"L{L}"] = row
        table[L] = row
        print(f"clean L{L}: " + "  ".join(f"{e} {row[e]['peak']:.4f}" for e in EST), flush=True)
    fits = {}
    sizes = np.array(sorted(table))
    models = {"a/L": lambda L, gc, a: gc + a / L,
              "a/L+b/L2": lambda L, gc, a, b: gc + a / L + b / L ** 2,
              "aL^-1/nu": lambda L, gc, a, inv_nu: gc + a * L ** (-inv_nu)}
    for minL in (32, 64):
        use = sizes[sizes >= minL]
        for e in EST:
            y = np.array([table[L][e]["peak"] for L in use])
            for name, fn in models.items():
                npar = fn.__code__.co_argcount - 1
                if use.size <= npar:
                    continue
                p0 = [1.0, -1.0] + ([1.0] if npar == 3 else [])
                try:
                    popt, pcov = curve_fit(fn, use.astype(float), y, p0=p0, maxfev=20000)
                    fits[f"Lmin{minL}_{e}_{name}"] = {"params": popt.tolist(), "errors": np.sqrt(np.diag(pcov)).tolist(), "sizes": use.tolist()}
                except RuntimeError:
                    fits[f"Lmin{minL}_{e}_{name}"] = None
    h = np.array([table[L]["xi"]["height"] for L in sizes])
    slope, icpt = np.polyfit(np.log(sizes), h, 1)
    fits["xi_height_vs_lnL"] = {"slope": float(slope), "intercept": float(icpt), "sizes": sizes.tolist(), "heights": h.tolist(),
                                "rmse": float(np.sqrt(np.mean((np.polyval([slope, icpt], np.log(sizes)) - h) ** 2)))}
    return res, fits


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clean, clean_fits = clean_part()
    rnd, rnd_fits = random_part()
    summary = {"estimators": EST, "clean": clean, "clean_fits": clean_fits, "random": rnd, "random_fits": rnd_fits}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
