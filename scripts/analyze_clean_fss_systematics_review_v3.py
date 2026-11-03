"""B5 (review v3, R1-7 and R2-MC2): systematics of the finite-size extrapolation of the clean chain.

1. Refines the pseudocritical points of Fig. 2 (clean OBC chain, h = 0, exact free fermions)
   beyond the 0.005 grid of the campaign:
   * xi_lambda from the nearest-neighbour bars 1 - C_{i,i+1} (the MST is the chain, Sec. III),
     C_{i,i+1} = -W[i+1, i], on a local grid of step 1e-4, maximum by a three-point parabola;
   * the physical response -d m2_bulk / dg (B = central half), central difference with
     h = 1e-4 and bounded scalar maximization (xatol 1e-7). m2_bulk uses Gaussian elimination
     restricted to the block B (same algorithm as cstag_elimination).
2. Fits g*(L) with several ansatzes and windows L >= Lmin, unweighted, as in the paper.
Output: results/review_v3/clean-fss-systematics-v1/summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clean_tfim_tn.exact.free_fermion_h0 import majorana_state  # noqa: E402

FIG2 = ROOT / "paper_figures/fig_clean_h0/output/fig_clean_h0_summary.json"
OUT = ROOT / "results/review_v3/clean-fss-systematics-v1"
SIZES = (16, 32, 64, 128, 256, 512)
STEP, HALF = 1e-4, 0.006
H = 1e-4


def W_of(L, g):
    return majorana_state(np.ones(L - 1), np.full(L, float(g)))[0]


def xi_curve(L, grid):
    bars = np.array([np.sort(1.0 - (-np.diagonal(W_of(L, g), -1))) for g in grid])
    d = np.gradient(bars, grid, axis=0, edge_order=2)
    return np.sqrt((bars * d ** 2).sum(-1) / (2 * (L - 1)))


def m2_bulk(L, g):
    W = W_of(L, g)
    M = -W[1:, :-1].T
    a, n = L // 4, L // 2           # B = sites a .. a+n-1 (0-based)
    tot = float(n)                  # diagonal
    for i in range(a, a + n - 1):
        A = M[i:a + n - 1, i:a + n - 1].copy()
        det = 1.0
        for k in range(A.shape[0]):
            p = A[k, k]
            det *= p
            tot += 2.0 * det
            if k + 1 < A.shape[0]:
                A[k + 1:, k + 1:] -= np.outer(A[k + 1:, k] / p, A[k, k + 1:])
    return tot / n ** 2


def parabola_peak(x, y):
    i = int(np.argmax(y))
    a, b, c = y[i - 1], y[i], y[i + 1]
    return float(x[i] + 0.5 * (a - c) / (a - 2 * b + c) * (x[1] - x[0]))


def refine(L, g0_xi, g0_phys):
    grid = np.arange(g0_xi - HALF, g0_xi + HALF + STEP / 2, STEP)
    gx = parabola_peak(grid, xi_curve(L, grid))
    f = lambda g: (m2_bulk(L, g + H) - m2_bulk(L, g - H)) / (2 * H)  # noqa: E731  (-d m2/dg is -f)
    r = minimize_scalar(f, bounds=(g0_phys - HALF, g0_phys + HALF), method="bounded", options={"xatol": 1e-7})
    return gx, float(r.x)


MODELS = {
    "a/L": (lambda L, gc, a: gc + a / L, 2),
    "a/L+b/L^2": (lambda L, gc, a, b: gc + a / L + b / L ** 2, 3),
    "a/L+c lnL/L^2": (lambda L, gc, a, c: gc + a / L + c * np.log(L) / L ** 2, 3),
    "(a+c lnL)/L": (lambda L, gc, a, c: gc + (a + c * np.log(L)) / L, 3),
    "a L^-x": (lambda L, gc, a, x: gc + a * L ** (-x), 3),
}


def fits(Ls, g):
    out = {}
    for name, (f, k) in MODELS.items():
        for lmin in (16, 32, 64, 128):
            m = Ls >= lmin
            if m.sum() < k + 1:
                continue
            p0 = [1.0, -1.0, 1.0][:k]
            try:
                p, cov = curve_fit(f, Ls[m], g[m], p0=p0, maxfev=20000)
            except RuntimeError:
                continue
            res = g[m] - f(Ls[m], *p)
            out[f"{name}|L>={lmin}"] = {"params": p.tolist(), "std": np.sqrt(np.diag(cov)).tolist(),
                                       "n": int(m.sum()), "rms_resid": float(np.sqrt(np.mean(res ** 2)))}
    for lmin in (16, 32, 64):  # g_c = 1 fixed, free exponent
        m = Ls >= lmin
        p, cov = curve_fit(lambda L, a, x: 1.0 + a * L ** (-x), Ls[m], g[m], p0=[-1.0, 1.0])
        out[f"gc=1, a L^-x|L>={lmin}"] = {"params": p.tolist(), "std": np.sqrt(np.diag(cov)).tolist(), "n": int(m.sum())}
    loc = np.log((1 - g[:-1]) / (1 - g[1:])) / np.log(Ls[1:] / Ls[:-1])
    out["local_inverse_nu_gc1"] = loc.tolist()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.loads(FIG2.read_text(encoding="utf-8"))["sizes"]
    peaks = {"xi": {}, "phys": {}, "xi_fig2": {}, "phys_fig2": {}}
    for L in SIZES:
        g0x, g0p = s[str(L)]["landscape_velocity_peak"]["x"], s[str(L)]["physical_peak"]["x"]
        gx, gp = refine(L, g0x, g0p)
        peaks["xi"][L], peaks["phys"][L] = gx, gp
        peaks["xi_fig2"][L], peaks["phys_fig2"][L] = g0x, g0p
        print(L, f"xi {g0x:.6f} -> {gx:.6f}   phys {g0p:.6f} -> {gp:.6f}", flush=True)
    Ls = np.array(SIZES, float)
    summary = {"peaks": {k: {str(L): v for L, v in d.items()} for k, d in peaks.items()}, "fits": {}}
    for key in ("xi", "phys", "xi_fig2", "phys_fig2"):
        summary["fits"][key] = fits(Ls, np.array([peaks[key][L] for L in SIZES]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
