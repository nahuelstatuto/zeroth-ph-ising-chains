"""B5 (review v3): refined maxima of every benchmark estimator for the clean chain at L = 256.

Table II gives g*(256) of each estimator from the 0.005 grid of the free-fermion campaign
(results/review_v1/free-fermion-h0-v1/clean_L256.npz) with the three-point parabola. For the
clean chain the peak of the sharper responses is comparable to that step, which biases the maxima
(documentation/RESULTADOS_B5_CLEAN_FSS_SYSTEMATICS_20261102.md). This script
  1. recomputes every curve on the campaign grid with the definitions of
     analyze_free_fermion_h0_review_v1.curves and checks that it reproduces the stored maxima;
  2. recomputes the curves on a local grid of step 1e-4 (+-0.006 around each maximum) and
     locates the maxima with the same parabola, and repeats with step 2e-4 as a convergence check.
Output: results/review_v3/clean-fss-systematics-v1/estimators_L256.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from clean_tfim_tn.exact.free_fermion_h0 import (cstag_elimination, log_fidelity_parity,  # noqa: E402
                                                 majorana_state, majorana_state_modes)
import analyze_free_fermion_h0_review_v1 as an  # noqa: E402
from run_free_fermion_h0_review_v1 import entropy_half  # noqa: E402

L = 256
OUT = ROOT / "results/review_v3/clean-fss-systematics-v1/estimators_L256.json"
KEYS = ("xi", "xi_nn", "phys", "dcnn", "fro", "d2e", "ds", "chiF")
B = slice(L // 4, L // 4 + L // 2)


def observables(grid, full_c=True):
    nn, m2, E, S, logF, C_all = [], [], [], [], [], []
    prev = None
    for g in grid:
        J, Gam = np.ones(L - 1), np.full(L, float(g))
        W, s, e = majorana_state(J, Gam)
        nn.append(-np.diagonal(W, -1))
        E.append(e)
        S.append(entropy_half(W))
        Wm, _, u, v = majorana_state_modes(J, Gam)
        if prev is not None:
            logF.append(log_fidelity_parity(prev, Wm, u, v))
        prev = Wm
        C = cstag_elimination(W) if full_c else None
        m2.append(C[B, B].mean())
        C_all.append(C)
    return np.array(nn), np.array(m2), np.array(E), np.array(S), np.array(logF), C_all


def curves(grid):
    nn, m2, E, S, logF, C_all = observables(grid)
    grad = lambda y: np.gradient(y, grid, axis=0, edge_order=2)  # noqa: E731
    bars = np.sort(np.clip(1.0 - nn, 0.0, 2.0), axis=-1)
    xi = np.sqrt((bars * grad(bars) ** 2).sum(-1) / (2 * bars.shape[-1]))
    fro = np.full(grid.size, np.nan)
    for k in range(1, grid.size - 1):
        fro[k] = np.linalg.norm(C_all[k + 1] - C_all[k - 1]) / (grid[k + 1] - grid[k - 1]) / L
    out = {"xi": xi, "xi_nn": xi, "phys": -grad(m2), "dcnn": np.abs(grad(nn.mean(-1))), "fro": fro,
           "d2e": -grad(grad(E / L)), "ds": np.abs(grad(S))}
    mid = 0.5 * (grid[1:] + grid[:-1])
    chi = -2.0 * logF / np.diff(grid) ** 2
    return out, mid, chi


def peak(x, y):
    p, edge = an.peaks(x, np.asarray(y)[None, :])
    if edge[0]:
        raise ValueError("maximum at the edge of the local grid")
    return float(p[0])


def main():
    z = np.load(an.DATA / "clean_L256.npz")
    grid = np.asarray(z["Gamma0"], float)
    stored = {k: v["peak"] for k, v in json.loads((an.DATA / "analysis" / "summary.json").read_text())["clean"]["L256"].items()
              if isinstance(v, dict) and "peak" in v}
    window = (grid > 0.95) & (grid < 1.05)          # the maxima of L = 256 all lie here
    gw = grid[window]
    out, mid, chi = curves(gw)
    campaign = {k: peak(gw, out[k]) for k in KEYS if k != "chiF"}
    campaign["chiF"] = peak(mid, chi)
    res = {"stored_campaign": stored, "recomputed_campaign_grid": campaign, "refined": {}, "refined_step2e-4": {}}
    for k in KEYS:
        print(f"{k:6s} stored {stored[k]:.6f} recomputed {campaign[k]:.6f}", flush=True)
    for step, tag in ((1e-4, "refined"), (2e-4, "refined_step2e-4")):
        for k in KEYS:
            if k == "xi_nn":
                continue
            g0 = campaign[k]
            fg = np.arange(g0 - 0.006, g0 + 0.006 + step / 2, step)
            o, m, c = curves(fg)
            res[tag][k] = peak(m, c) if k == "chiF" else peak(fg[1:-1], o[k][1:-1]) if k == "fro" else peak(fg, o[k])
            print(f"{tag} {k:6s} {res[tag][k]:.6f}", flush=True)
        res[tag]["xi_nn"] = res[tag]["xi"]
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
