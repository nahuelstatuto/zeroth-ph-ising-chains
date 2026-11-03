"""Reentrant OBC cohorts with odd L (L17 h=2.5, L33 h=2.3125; 400 seeds each).

Answers the questions frozen in documentation/PLAN_ODD_L_OBC_REVIEW_TESTS_20261014.md
with the same rules used for L16/L32 (scripts/analyze_reentrant_review_checks.py and
paper_figures/fig_reentrant/common.py), so that the two parities can be compared:

  S. frozen physical selection on C(R), R = floor(L/2) - 1 (7 and 15, odd distances);
  A. exact classical ground state at Gamma0 = 0 vs DMRG, domain-wall counts (parity);
  B. entry anatomy on the selected seeds: Delta sum<sigma^z> (saved MPS) and NN walls;
  C. entry/exit localization: physical slope, pair L1/L2, long-range landscape w=0,
     H0 global velocity (split at the apex of C(R)), frozen H0 entry rule; crossing slopes;
  D. exit vs maximum of the half-chain entropy after the entry;
  E. disorder-averaged hump of odd C(R) and even-distance control;
  F. exit vs the clean boundary at mean couplings.

Read-only, no DMRG. Output:
results/random_ising_gamma_ensemble/reentrant-oddL-L17-L33-v1/analysis/{summary.json, selection_L*.csv, cache_L*.npz}
"""

import csv
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "paper_figures" / "fig_reentrant"))
from analyze_pbc_reentrant_selected_seed_topology import select_physical_seeds  # noqa: E402
from analyze_obc_L32_all1000_complete import sign_class  # noqa: E402
from analyze_reentrant_review_checks import classical_gs_obc, stagger_r, mae, split, REF_H, REF_G  # noqa: E402
from analyze_reentrant_exit_anatomy import mst_deaths  # noqa: E402
import common as cm  # noqa: E402

BASE = ROOT / "results" / "random_ising_gamma_ensemble"
OUT = BASE / "reentrant-oddL-L17-L33-v1" / "analysis"
COHORTS = {17: ("L17-reentrant-h2500", 2.5), 33: ("L33-reentrant-h23125", 2.3125)}
SELECTION = {"endpoint_points": 2, "negative_margin": 0.02, "positive_amplitude": 0.05,
             "minimum_lobe_width": 0.10, "minimum_positive_run": 2}
GRID = np.linspace(0.0, 2.0, 401)
RNG = np.random.default_rng(20261016)
WORKERS = 6


def load(L, prefix):
    rows = []
    for st in sorted(glob.glob(str(BASE / f"{prefix}-oddL-*" / "campaign_status.json"))):
        for j in json.loads(Path(st).read_text(encoding="utf-8"))["jobs"]:
            if j["status"] != "complete":
                raise RuntimeError(f"incomplete job in {st}")
            rows.append((int(j["seed"]), j["target_directory"]))
    rows.sort()
    if len({s for s, _ in rows}) != len(rows):
        raise RuntimeError("duplicate seeds")
    data = {"seeds": [], "dirs": [], "C": [], "J": [], "S": [], "flag": []}
    for seed, d in rows:
        with np.load(Path(d) / "continuation_summary.npz") as z:
            g = np.asarray(z["Gamma0_values"], float)
            data["C"].append(np.asarray(z["Cstag"], np.float32))
            data["J"].append(np.asarray(z["J"], float))
            data["S"].append(np.asarray(z["half_chain_entropy"], float))
            data["flag"].append(np.asarray(z["branch_disagreement"], bool))
        data["seeds"].append(seed)
        data["dirs"].append(d)
    out = {k: np.asarray(v) for k, v in data.items() if k != "dirs"}
    out["dirs"], out["g"] = data["dirs"], g
    return out


def landscape_velocity(deaths_stack, step):
    """L2 velocity of the H0 landscape between consecutive fields (frozen forward-difference form)."""
    tents = np.maximum(np.minimum(GRID[None, None, :], deaths_stack[:, :, None] - GRID[None, None, :]), 0.0)
    layers = np.sort(tents, axis=1)[:, ::-1, :]
    d = np.diff(layers, axis=0) / step
    return np.sqrt(np.trapezoid(np.sum(d ** 2, axis=1), GRID, axis=1) / deaths_stack.shape[1])


def shell_deaths(M, edges):
    L = M.shape[0]
    w = 1.0 - M[edges[:, 0], edges[:, 1]]
    parent = list(range(L))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    out = []
    for k in np.argsort(w, kind="stable"):
        a, b = map(int, edges[k])
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            out.append(float(w[k]))
    return np.sort(out)[::-1]


def measure_mz(scan_dir):
    from clean_tfim_tn.io.mps_hdf5 import load_mps_collection_hdf5

    scan_dir = Path(scan_dir)
    manifest = json.loads((scan_dir / "selected_mps_manifest.json").read_text(encoding="utf-8"))
    states = load_mps_collection_hdf5(scan_dir / manifest["mps_file"])["selected_states"]
    return np.asarray([np.real(states[r["state_key"]].expectation_value("Sigmaz")) for r in manifest["point_records"]])


def cohort(L, prefix, h):
    d = load(L, prefix)
    g, C = d["g"], d["C"]
    step = g[1] - g[0]
    mid = 0.5 * (g[1:] + g[:-1])
    R = L // 2 - 1
    cR = np.diagonal(C, offset=R, axis1=-2, axis2=-1).mean(-1).astype(float)
    rows, sel = select_physical_seeds(g, d["seeds"], cR, **SELECTION)
    sel = np.asarray(sel, int)
    classes = [sign_class(c) for c in cR]
    res = {"h": h, "R": R, "seeds": int(len(d["seeds"])), "grid_points": int(g.size), "step": float(step)}
    res["S_selection"] = {"selected": int(sel.size), "fraction": float(sel.size / len(d["seeds"])),
                          "sign_classes": {k: classes.count(k) for k in sorted(set(classes))},
                          "fraction_C_positive_at_Gamma0_0": float(np.mean(cR[:, 0] > 0))}
    with (OUT / f"selection_L{L}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # A. classical ground state at Gamma0 = 0 and wall parity
    cls_C, cls_w, dmrg_w = [], [], []
    for J, Cn in zip(d["J"], C):
        s = classical_gs_obc(J, h)
        cls_C.append(stagger_r(s, R))
        cls_w.append(int(np.sum(s[:-1] * s[1:] == 1)))
        dmrg_w.append(int(np.sum(np.diagonal(Cn[0], offset=1) < -0.5)))
    cls_C, cls_w = np.array(cls_C), np.array(cls_w)
    hist = lambda a: {int(k): int(v) for k, v in zip(*np.unique(a, return_counts=True))}
    res["A_classical_Gamma0_0"] = {"classical_mean_C": float(cls_C.mean()), "dmrg_mean_C": float(cR[:, 0].mean()),
                                   "max_abs_classical_minus_dmrg": float(np.abs(cls_C - cR[:, 0]).max()),
                                   "classical_wall_count_hist": hist(cls_w), "dmrg_wall_count_hist": hist(dmrg_w),
                                   "fraction_without_walls": float(np.mean(cls_w == 0))}

    entry = np.array([rows[i]["entry_crossing"] for i in sel], float)
    exit_ = np.array([rows[i]["exit_crossing"] for i in sel], float)
    Csel = C[sel].astype(float)

    # B. entry anatomy with magnetizations measured on the saved MPS
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        mz = np.stack(list(pool.map(measure_mz, [d["dirs"][i] for i in sel])))
    dM, wb, wa = [], [], []
    for n in range(sel.size):
        ib = int(np.searchsorted(g, entry[n]) - 1)
        ia = ib + 1
        dM.append(mz[n, ia].sum() - mz[n, ib].sum())
        wb.append(int(np.sum(np.diagonal(Csel[n, ib], offset=1) < -0.5)))
        wa.append(int(np.sum(np.diagonal(Csel[n, ia], offset=1) < -0.5)))
    dM = np.array(dM)
    res["B_entry_anatomy"] = {"delta_M_median": float(np.median(dM)),
                              "delta_M_quartiles": [float(v) for v in np.percentile(dM, [25, 75])],
                              "fraction_dM_near_0": float(np.mean(np.abs(dM) < 0.5)),
                              "fraction_dM_near_minus2": float(np.mean(np.abs(dM + 2) < 0.5)),
                              "walls_before_hist": hist(np.array(wb)), "walls_after_hist": hist(np.array(wa))}

    # C. localization of entry and exit
    vel_h0 = np.stack([landscape_velocity(np.stack([mst_deaths(Csel[n, i]) for i in range(g.size)]), step)
                       for n in range(sel.size)])
    edges = np.asarray([(i, j) for i in range(L) for j in range(i + 1, L) if j - i == R], int)
    vel_lr = np.stack([landscape_velocity(np.stack([shell_deaths(Csel[n, i], edges) for i in range(g.size)]), step)
                       for n in range(sel.size)])
    pairs = np.diagonal(Csel, offset=R, axis1=-2, axis2=-1)
    dpairs = np.diff(pairs, axis=1) / step
    cRs = cR[sel]
    base = {"phys_abs_slope": np.abs(np.diff(cRs, axis=1) / step),
            "phys_L1_pairs": np.abs(dpairs).mean(-1), "phys_L2_pairs": np.sqrt((dpairs ** 2).mean(-1)),
            "LR_landscape_w0": vel_lr, "H0_global": vel_h0}
    res["C_entry_MAE"], res["C_exit_MAE"] = {}, {}
    for name, v in base.items():
        en, ex = split(v, cRs, g, mid, step)
        res["C_entry_MAE"][name] = mae(en, entry, step)
        res["C_exit_MAE"][name] = mae(ex, exit_, step)
    fake = {"H0_velocity": vel_h0, "Gamma0_mid": mid, "Gamma0": g}
    res["C_entry_MAE"]["H0_global_frozen_rule"] = mae(cm.topological_entry(fake), entry, step)
    res["C_exit_MAE"]["H0_second_wave"] = mae(cm.second_wave(fake)[0], exit_, step)
    res["C_entry_MAE"]["constant"] = mae(np.full_like(entry, np.median(entry)), entry, step)
    res["C_exit_MAE"]["constant"] = mae(np.full_like(exit_, np.median(exit_)), exit_, step)
    slopes_e, slopes_x = [], []
    for c in cRs:
        (e, se), (x, sx) = cm.entry_exit(g, c)
        slopes_e.append(se)
        slopes_x.append(abs(sx))
    res["C_crossing_slopes"] = {"entry_median": float(np.nanmedian(slopes_e)), "exit_median": float(np.nanmedian(slopes_x)),
                                "entry_quartiles": [float(v) for v in np.nanpercentile(slopes_e, [25, 75])],
                                "exit_quartiles": [float(v) for v in np.nanpercentile(slopes_x, [25, 75])]}

    # D. independent exit: maximum of the half-chain entropy after the entry
    S_half = d["S"][sel]
    s_exit = np.array([g[(g > e)][np.argmax(s[(g > e)])] for s, e in zip(S_half, entry)])
    interior = s_exit < g[-1] - 1e-9
    _, lr_ex = split(vel_lr, cRs, g, mid, step)
    res["D_entropy_exit"] = {"fraction_interior_maximum": float(interior.mean()),
                             "entropy_max_vs_C_zero_exit": mae(np.where(interior, s_exit, np.nan), exit_, step),
                             "LR_w0_velocity_vs_entropy_exit": mae(np.where(interior, lr_ex, np.nan),
                                                                   np.where(interior, s_exit, np.nan), step)}

    # E. disorder-averaged hump (all seeds), odd C(R) vs even control
    odd = cR
    even = 0.5 * (np.diagonal(C, offset=R - 1, axis1=-2, axis2=-1).mean(-1)
                  + np.diagonal(C, offset=R + 1, axis1=-2, axis2=-1).mean(-1)).astype(float)
    k = int(np.argmax(odd.mean(0)))
    draws = RNG.integers(0, odd.shape[0], (4000, odd.shape[0]))
    d_odd = (odd[:, k] - odd[:, 0])[draws].mean(1)
    d_even = (even[:, k] - even[:, 0])[draws].mean(1)
    res["E_average_hump"] = {"Gamma_star": float(g[k]), "odd_increase": float(d_odd.mean()),
                             "odd_increase_ci95": [float(v) for v in np.percentile(d_odd, [2.5, 97.5])],
                             "even_increase": float(d_even.mean()),
                             "even_increase_ci95": [float(v) for v in np.percentile(d_even, [2.5, 97.5])],
                             "odd_mean_curve": odd.mean(0).round(4).tolist()}

    # F. exit vs the clean boundary at mean couplings
    pred = float(np.interp(h / 1.5, REF_H, REF_G))
    res["F_clean_boundary"] = {"clean_Gamma_over_J": pred, "median_exit": float(np.median(exit_)),
                               "median_entry": float(np.median(entry)),
                               "MAE_exit_vs_clean": float(np.mean(np.abs(exit_ - pred)))}
    np.savez_compressed(OUT / f"cache_L{L}.npz", seeds=d["seeds"], Gamma0=g, curves_R=cR, selected=sel,
                        entry=entry, exit=exit_, mz=mz, H0_velocity=vel_h0, LR_velocity_w0=vel_lr)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for L, (prefix, h) in COHORTS.items():
        out[f"L{L}"] = cohort(L, prefix, h)
        print(L, "done", flush=True)
    even = json.loads((BASE / "reentrant-exit-anatomy-v1" / "analysis" / "review_checks.json").read_text(encoding="utf-8"))
    out["even_reference"] = {k: {kk: even[k][kk] for kk in ("A_classical_Gamma0_0", "B_entry_anatomy", "C_entry_MAE",
                                                            "C_exit_MAE", "D_entropy_exit", "F_clean_boundary")}
                             for k in ("L16", "L32")}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
