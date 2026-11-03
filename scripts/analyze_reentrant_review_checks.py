"""Checks requested by the internal review of the reentrant section (2026-10-14).

Read-only on existing caches (no DMRG). OBC, fixed_mean, C=Cstag raw.
  A. exact classical ground state (Gamma0=0) for our own J realizations vs DMRG at Gamma0=0;
  B. entry anatomy on the selected seeds: Delta sum<sigma^z> and NN walls before/after entry;
  C. non-topological baselines for the w=0 circularity: argmax|dC(r)/dGamma0|, L1/L2 of dC_i/dGamma0;
  D. exit with an independent physical observable: maximum of the half-chain entropy after entry;
  E. disorder average: paired bootstrap of the hump of C(odd r) and the even-distance control;
  F. exit vs the clean boundary (digitized Lajko Fig. 1) at mean couplings (Gamma_bar/J_bar = Gamma0).
Output: results/random_ising_gamma_ensemble/reentrant-exit-anatomy-v1/analysis/review_checks.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "results" / "random_ising_gamma_ensemble" / "reentrant-exit-anatomy-v1" / "analysis"
FIG = ROOT / "paper_figures" / "fig_reentrant"
sys.path.insert(0, str(FIG))
import common as cm  # noqa: E402

H = {16: 2.5, 32: 2.3125}
REF_H = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.9, 1.95, 2.0])
REF_G = np.array([1.0, 0.992, 0.970, 0.931, 0.878, 0.802, 0.712, 0.599, 0.453, 0.278, 0.154, 0.077, 0.0])
RNG = np.random.default_rng(20261014)
S = np.array([1, -1])


def classical_gs_obc(J, h):
    """Exact Gamma=0 ground state of sum J s s - h sum s (OBC) by dynamic programming."""
    L = len(J) + 1
    E = -h * S.astype(float)
    back = np.zeros((L, 2), int)
    for i in range(1, L):
        cand = E[:, None] + J[i - 1] * S[:, None] * S[None, :] - h * S[None, :]
        back[i] = np.argmin(cand, axis=0)
        E = cand[back[i], [0, 1]]
    k = int(np.argmin(E))
    conf = [k]
    for i in range(L - 1, 0, -1):
        k = back[i][k]
        conf.append(k)
    return S[np.array(conf[::-1])]


def stagger_r(s, r):
    return (-1) ** r * np.mean(s[:-r] * s[r:])


def mae(est, ref, step):
    ok = np.isfinite(est)
    ae = np.abs(est[ok] - ref[ok])
    boot = ae[RNG.integers(0, ae.size, (4000, ae.size))].mean(1)
    return {"N": int(ok.sum()), "of": int(ref.size), "MAE": float(ae.mean()),
            "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
            "within_one_step": float(np.mean(ae <= step + 1e-9))}


def split(values, level, g, mid, step):
    """Maximum of values (on midpoints) left / right of the apex of level (frozen LR rule)."""
    first = 2 * step
    en, ex = [], []
    for v, m in zip(values, level):
        el = np.flatnonzero(g >= first - 1e-12)
        apex = g[el[np.argmax(m[el])]]
        left = np.flatnonzero((mid >= first + step / 2 - 1e-12) & (mid <= apex + 1e-12))
        right = np.flatnonzero(mid > apex + 1e-12)
        en.append(mid[left[np.argmax(v[left])]] if left.size else np.nan)
        ex.append(mid[right[np.argmax(v[right])]] if right.size else np.nan)
    return np.array(en), np.array(ex)


def main():
    out = {}
    for L in (16, 32):
        cache = np.load(ANA / f"cache_L{L}.npz")
        mag = np.load(ANA / f"local_magnetization_L{L}.npz")
        fig = np.load(FIG / "data_raw" / f"reentrant_L{L}.npz")
        g = cache["Gamma0"]
        step = g[1] - g[0]
        mid = 0.5 * (g[1:] + g[:-1])
        R = L // 2 - 1
        h = H[L]
        C = cache["Cstag"]                                           # (1000, G, L, L) float32
        res = {"h": h, "R": R}

        # A. classical ground state vs DMRG at Gamma0 = 0
        cls_C, cls_w, weakest = [], [], []
        for J in cache["J"]:
            s = classical_gs_obc(J, h)
            walls = np.flatnonzero(s[:-1] * s[1:] == 1)
            cls_C.append(stagger_r(s, R))
            cls_w.append(walls.size)
            if walls.size == 1:
                odd_bonds = np.arange(0, L - 1, 2)                  # bonds whose Neel ends are parallel when flipped
                weakest.append(int(walls[0] == odd_bonds[np.argmin(J[odd_bonds])]))
        cls_C, cls_w = np.array(cls_C), np.array(cls_w)
        dmrg_C0 = np.diagonal(C[:, 0], offset=R, axis1=-2, axis2=-1).mean(-1).astype(float)
        dmrg_w0 = (np.diagonal(C[:, 0], offset=1, axis1=-2, axis2=-1) < -0.5).sum(-1)
        res["A_classical_Gamma0_0"] = {
            "classical_mean_C": float(cls_C.mean()), "classical_sem": float(cls_C.std() / np.sqrt(cls_C.size)),
            "dmrg_mean_C_Gamma0_0": float(dmrg_C0.mean()),
            "max_abs_classical_minus_dmrg": float(np.abs(cls_C - dmrg_C0).max()),
            "fraction_seeds_equal": float(np.mean(np.abs(cls_C - dmrg_C0) < 1e-3)),
            "classical_wall_count_hist": {int(k): int(v) for k, v in zip(*np.unique(cls_w, return_counts=True))},
            "dmrg_wall_count_hist": {int(k): int(v) for k, v in zip(*np.unique(dmrg_w0, return_counts=True))},
            "fraction_without_walls": float(np.mean(cls_w == 0)),
            "single_wall_on_weakest_odd_bond": float(np.mean(weakest)) if weakest else None}

        # B. entry anatomy (selected seeds)
        sel = np.flatnonzero(cache["selected"])
        entry, exit_ = cache["entry"][sel], cache["exit"][sel]
        mz = mag["mz"]
        dM, w_before, w_after, edge_before, edge_after = [], [], [], [], []
        for n, si in enumerate(sel):
            ib = int(np.searchsorted(g, entry[n]) - 1)
            ia = ib + 1
            dM.append(mz[n, ia].sum() - mz[n, ib].sum())
            for idx, wl, ed in ((ib, w_before, edge_before), (ia, w_after, edge_after)):
                nn = np.diagonal(C[si, idx], offset=1).astype(float)
                walls = np.flatnonzero(nn < -0.5)
                wl.append(walls.size)
                ed.append(int(np.min(np.minimum(walls, L - 2 - walls))) if walls.size else -1)
        dM = np.array(dM)
        res["B_entry_anatomy"] = {
            "delta_M_median": float(np.median(dM)), "delta_M_quartiles": [float(v) for v in np.percentile(dM, [25, 75])],
            "fraction_dM_near_0": float(np.mean(np.abs(dM) < 0.5)), "fraction_dM_near_minus2": float(np.mean(np.abs(dM + 2) < 0.5)),
            "walls_before_hist": {int(k): int(v) for k, v in zip(*np.unique(w_before, return_counts=True))},
            "walls_after_hist": {int(k): int(v) for k, v in zip(*np.unique(w_after, return_counts=True))},
            "edge_distance_before_median": float(np.median([e for e in edge_before if e >= 0])) if any(e >= 0 for e in edge_before) else None,
            "edge_distance_after_hist": {int(k): int(v) for k, v in zip(*np.unique(edge_after, return_counts=True))}}

        # C. baselines for the w = 0 circularity
        Csel = C[sel].astype(float)
        pairs = np.diagonal(Csel, offset=R, axis1=-2, axis2=-1)          # (n, G, L-R)
        cR = pairs.mean(-1)
        dpairs = np.diff(pairs, axis=1) / step
        base = {"phys_abs_slope": np.abs(np.diff(cR, axis=1) / step),
                "phys_L1_pairs": np.abs(dpairs).mean(-1),
                "phys_L2_pairs": np.sqrt((dpairs ** 2).mean(-1)),
                "LR_landscape_w0": fig["LR_velocity_w0"].astype(float),
                "H0_global": fig["H0_velocity"].astype(float)}
        res["C_entry_MAE"], res["C_exit_MAE"] = {}, {}
        for name, v in base.items():
            en, ex = split(v, cR, g, mid, step)
            res["C_entry_MAE"][name] = mae(en, entry, step)
            res["C_exit_MAE"][name] = mae(ex, exit_, step)
        res["C_entry_MAE"]["H0_global_frozen_rule"] = mae(cm.topological_entry(fig), entry, step)
        res["C_entry_MAE"]["constant"] = mae(np.full_like(entry, np.median(entry)), entry, step)
        res["C_exit_MAE"]["constant"] = mae(np.full_like(exit_, np.median(exit_)), exit_, step)

        # D. independent exit: maximum of the half-chain entropy after the entry
        S_half = cache["half_chain_entropy"][sel]
        s_exit = np.array([g[(g > e)][np.argmax(s[(g > e)])] for s, e in zip(S_half, entry)])
        interior = (s_exit < g[-1] - 1e-9)
        lr_en, lr_ex = split(base["LR_landscape_w0"], cR, g, mid, step)
        res["D_entropy_exit"] = {
            "fraction_interior_maximum": float(interior.mean()),
            "entropy_max_vs_C_zero_exit": mae(np.where(interior, s_exit, np.nan), exit_, step),
            "LR_w0_velocity_vs_entropy_exit": mae(np.where(interior, lr_ex, np.nan), np.where(interior, s_exit, np.nan), step),
            "H0_second_wave_vs_entropy_exit": mae(np.where(interior, cm.second_wave(fig)[0], np.nan), np.where(interior, s_exit, np.nan), step),
            "median_entropy_exit": float(np.median(s_exit[interior])), "median_C_exit": float(np.median(exit_))}

        # E. disorder average: hump of odd C(r) and even-distance control, paired bootstrap over all 1000 seeds
        odd = np.diagonal(C, offset=R, axis1=-2, axis2=-1).mean(-1).astype(float)
        even = 0.5 * (np.diagonal(C, offset=R - 1, axis1=-2, axis2=-1).mean(-1)
                      + np.diagonal(C, offset=R + 1, axis1=-2, axis2=-1).mean(-1)).astype(float)
        k = int(np.argmax(odd.mean(0)))
        draws = RNG.integers(0, odd.shape[0], (4000, odd.shape[0]))
        d_odd = (odd[:, k] - odd[:, 0])[draws].mean(1)
        d_even = (even[:, k] - even[:, 0])[draws].mean(1)
        res["E_average_hump"] = {
            "Gamma_star": float(g[k]), "odd_mean_curve": odd.mean(0).round(4).tolist(), "even_mean_curve": even.mean(0).round(4).tolist(),
            "odd_increase": float(d_odd.mean()), "odd_increase_ci95": [float(v) for v in np.percentile(d_odd, [2.5, 97.5])],
            "even_increase": float(d_even.mean()), "even_increase_ci95": [float(v) for v in np.percentile(d_even, [2.5, 97.5])],
            "note": "Gamma_star chosen as the argmax of the odd average (selection bias toward a positive increase)"}

        # F. exit vs clean boundary at mean couplings
        pred = float(np.interp(h / 1.5, REF_H, REF_G))
        res["F_clean_boundary"] = {"h_over_Jbar": h / 1.5, "clean_Gamma_over_J": pred,
                                   "median_exit": float(np.median(exit_)), "MAE_exit_vs_clean": float(np.mean(np.abs(exit_ - pred))),
                                   "median_entry": float(np.median(entry))}
        out[f"L{L}"] = res
        print(L, "done", flush=True)
    (ANA / "review_checks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
