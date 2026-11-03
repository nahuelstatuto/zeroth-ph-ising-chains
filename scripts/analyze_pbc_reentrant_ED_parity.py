"""PBC ED parity campaign (L16 vs L17): questions frozen in
documentation/PLAN_PBC_ED_PARITY_L16_L17_20261023.md, answered with the OBC rules unchanged.

  P. physical, main cohort (seeds 5000-5999, h = 2.5), all seeds: disorder-averaged C(R), R = 7,
     bootstrap band, zero crossings, fraction of resamples with the negative-positive-negative
     pattern; classical wall count at Gamma0 = 0 and C(R) per wall count; sign classes; frozen
     per-seed selection; even-distance control (R +/- 1) of the average hump;
  M. h map (seeds 6000-6199, h = 2.40-2.60, paired across h): finite-size boundary h*_L(Gamma0)
     from the zero of the average after isotonic decreasing regression in h (Lajko et al.
     criterion), bootstrap over seeds, lobe height h*(Gamma0) - h*(0);
  T. topology on the selected seeds of the main cohort: H0 global velocity (frozen entry rule,
     split at the apex of C(R), second wave), long-range landscape w = 0 and w = 1 with circular
     distances, H1 growth after the entry (tau = 0.04), constant; MAE against the physical
     crossings, next to the OBC L16 / L17 values already published in the repository.

PBC differences that matter for reading T: the w = 0 shell {(i, i+R mod L)} is one L-cycle
(gcd(R, L) = 1), so its Kruskal forest drops one edge and the OBC identity M_0 = C(R) holds only
up to that edge; C is Cstag with the circular phase (-1)^min(d, L-d).

Read-only. Output: results/random_ising_gamma_ensemble/pbc-reentrant-ED-parity-v1/analysis/
Run:  python scripts/analyze_pbc_reentrant_ED_parity.py            (all parts)
      python scripts/analyze_pbc_reentrant_ED_parity.py --partial  (QC on the seeds finished so far)
"""

import argparse
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
from analyze_reentrant_review_checks import mae, split  # noqa: E402
from analyze_reentrant_exit_anatomy import mst_deaths  # noqa: E402
from analyze_reentrant_oddL_L17_L33 import SELECTION, landscape_velocity, shell_deaths  # noqa: E402
from analyze_clean_L64_new_topology import h1_diagram  # noqa: E402
from run_pbc_reentrant_ED_parity import SELECTION as RUN_SELECTION, GRID as RUN_GRID, c_of_r  # noqa: E402
import common as cm  # noqa: E402

BASE = ROOT / "results" / "random_ising_gamma_ensemble"
OUT = BASE / "pbc-reentrant-ED-parity-v1" / "analysis"
SPEC = ROOT / "campaign_specs" / "pbc_reentrant_ED_parity_L16_L17_v1.json"
OBC_EVEN = BASE / "reentrant-exit-anatomy-v1" / "analysis" / "review_checks.json"
OBC_ODD = BASE / "reentrant-oddL-L17-L33-v1" / "analysis" / "summary.json"
H1_TAU = 0.04
LOBE_GAMMA0 = 0.225          # fixed a priori: maximum of the PBC L16 lobe (DMRG 2026-09-13, ED review 2026-10-19)
N_BOOT = 4000
RNG = np.random.default_rng(20261023)
WORKERS = 6


def load(L, h, seeds, partial):
    """C, J and QC of one (L, h) cohort from all its blocks; refuses duplicates and gaps."""
    data = {}
    for status_path in sorted(BASE.glob(f"pbc-reentrant-ED-parity-L{L}-h{round(h * 10000):05d}-s*-v1/campaign_status.json")):
        manifest = json.loads((status_path.parent / "manifest.json").read_text(encoding="utf-8"))
        if manifest["selection"] != SELECTION or not np.allclose(manifest["Gamma0"], RUN_GRID):
            raise ValueError(f"frozen setup changed in {status_path.parent}")
        for job in json.loads(status_path.read_text(encoding="utf-8"))["jobs"]:
            s = int(job["seed"])
            if s not in seeds or job["status"] != "complete":
                continue
            if s in data:
                raise RuntimeError(f"seed {s} appears in two blocks")
            npz = status_path.parent / "seeds" / f"scan_s{s:04d}.npz"
            info = json.loads(npz.with_suffix(".json").read_text(encoding="utf-8"))
            with np.load(npz) as z:
                data[s] = (np.asarray(z["C"], float), np.asarray(z["J"], float), np.asarray(z["E1"] - z["E0"]),
                           np.asarray(z["half_ring_entropy"]), info)
    missing = sorted(set(seeds) - set(data))
    if missing and not partial:
        raise RuntimeError(f"L{L} h={h}: {len(missing)} seeds missing (first {missing[:5]}); use --partial for QC")
    keys = sorted(data)
    if not keys:
        return None
    return {"seeds": np.array(keys), "C": np.stack([data[s][0] for s in keys]), "J": np.stack([data[s][1] for s in keys]),
            "gap": np.stack([data[s][2] for s in keys]), "S": np.stack([data[s][3] for s in keys]),
            "info": [data[s][4] for s in keys], "missing": len(missing), "g": RUN_GRID.copy()}


def crossings(g, y):
    s = np.flatnonzero(np.sign(y[:-1]) != np.sign(y[1:]))
    return [float(g[i] - y[i] * (g[i + 1] - g[i]) / (y[i + 1] - y[i])) for i in s]


def physical(d, L):
    g, C, R = d["g"], d["C"], L // 2 - 1
    cR = c_of_r(C, R)
    n = cR.shape[0]
    draws = RNG.integers(0, n, (N_BOOT, n))
    boot = cR[draws].mean(1)
    pattern = np.array([sign_class(b) for b in boot])
    walls0 = np.array([info["classical_parallel_bonds_Gamma0_0"] for info in d["info"]])
    by_walls = {int(k): {"count": int(np.sum(walls0 == k)), "mean_C_R_Gamma0_0": float(cR[walls0 == k, 0].mean())}
                for k in np.unique(walls0)}
    rows, sel = select_physical_seeds(g, d["seeds"], cR, **SELECTION)
    classes = [sign_class(c) for c in cR]
    i = np.arange(L)
    even = 0.5 * (C[..., i, (i + R - 1) % L].mean(-1) + C[..., i, (i + R + 1) % L].mean(-1))
    k = int(np.argmax(cR.mean(0)))
    gap = d["gap"][:, 1:]
    res = {"seeds": int(n), "missing": d["missing"], "R": R,
           "mean_curve": cR.mean(0).round(5).tolist(),
           "band95": np.percentile(boot, [2.5, 97.5], axis=0).round(5).tolist(),
           "mean_at_0": float(cR[:, 0].mean()), "max": float(cR.mean(0).max()), "argmax": float(g[k]),
           "zero_crossings_of_mean": crossings(g, cR.mean(0)),
           "bootstrap_sign_pattern": {c: float(np.mean(pattern == c)) for c in sorted(set(pattern))},
           "classical_walls_Gamma0_0": by_walls,
           "fraction_C_R_positive_Gamma0_0": float(np.mean(cR[:, 0] > 0)),
           "sign_classes": {c: classes.count(c) for c in sorted(set(classes))},
           "selected": int(len(sel)), "selected_fraction": float(len(sel) / n),
           "hump_odd_increase": float((cR[:, k] - cR[:, 0]).mean()),
           "hump_odd_increase_ci95": np.percentile((cR[:, k] - cR[:, 0])[draws].mean(1), [2.5, 97.5]).tolist(),
           "hump_even_increase": float((even[:, k] - even[:, 0]).mean()),
           "hump_even_increase_ci95": np.percentile((even[:, k] - even[:, 0])[draws].mean(1), [2.5, 97.5]).tolist(),
           "QC": {"max_residual_E0": max(x["max_residual_E0"] for x in d["info"]),
                  "eigsh_retries": int(sum(x["eigsh_retries"] for x in d["info"])),
                  "near_degenerate_points_gap_lt_1e-8": int(np.sum(gap < 1e-8)),
                  "seeds_with_near_degenerate_points": int(np.sum((gap < 1e-8).any(1))),
                  "classical_degenerate_seeds_Gamma0_0": int(sum(x["classical_degeneracy_Gamma0_0"] > 1 for x in d["info"]))}}
    return res, cR, rows, np.asarray(sel, int)


def isotonic_decreasing(y):
    """Pool-adjacent-violators for a non-increasing fit (equal weights)."""
    blocks = [[v, 1] for v in -np.asarray(y, float)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] / blocks[i][1] > blocks[i + 1][0] / blocks[i + 1][1]:
            blocks[i] = [blocks[i][0] + blocks[i + 1][0], blocks[i][1] + blocks[i + 1][1]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return -np.concatenate([[b[0] / b[1]] * b[1] for b in blocks])


def h_star(hs, curve_by_h):
    """Zero of the isotonic (decreasing in h) average, linear interpolation; nan if not bracketed."""
    y = isotonic_decreasing(curve_by_h)
    for i in range(len(hs) - 1):
        if y[i] > 0 >= y[i + 1]:
            return float(hs[i] + y[i] * (hs[i + 1] - hs[i]) / (y[i] - y[i + 1]))
    return np.nan


def h_map(L, spec, partial):
    c = spec["cohorts"]["h_map"]
    hs = np.array(c["h"])
    seeds = set(range(c["seed_range"][0], c["seed_range"][1] + 1))
    cohorts = [load(L, h, seeds, partial) for h in hs]
    if any(x is None for x in cohorts):
        return None
    common = sorted(set.intersection(*[set(x["seeds"].tolist()) for x in cohorts]))
    R = L // 2 - 1
    cR = np.stack([c_of_r(x["C"][np.isin(x["seeds"], common)], R) for x in cohorts], axis=1)   # (seed, h, Gamma0)
    g = RUN_GRID
    central = np.array([h_star(hs, cR[:, :, k].mean(0)) for k in range(g.size)])
    draws = RNG.integers(0, len(common), (N_BOOT, len(common)))
    boot = np.array([[h_star(hs, cR[dr, :, k].mean(0)) for k in range(g.size)] for dr in draws[:1000]])
    lobe = boot - boot[:, :1]
    k_max = int(np.nanargmax(central)) if np.isfinite(central).any() else 0
    k_fix = int(np.argmin(np.abs(g - LOBE_GAMMA0)))
    fixed = lobe[:, k_fix]
    return {"seeds_paired": len(common), "h": hs.tolist(), "h_star_central": central.round(5).tolist(),
            "h_star_bracketed_fraction": np.mean(np.isfinite(boot), axis=0).round(3).tolist(),
            "primary_lobe_Gamma0": LOBE_GAMMA0, "primary_lobe_height": float(central[k_fix] - central[0]),
            "primary_lobe_ci95": np.percentile(fixed[np.isfinite(fixed)], [2.5, 97.5]).tolist()
            if np.isfinite(fixed).sum() > 20 else None,
            "primary_lobe_bracketed_fraction": float(np.mean(np.isfinite(fixed))),
            "Gamma0_of_max": float(g[k_max]), "lobe_height_central": float(central[k_max] - central[0]),
            "lobe_height_ci95_at_max": np.nanpercentile(lobe[:, k_max], [2.5, 97.5]).tolist()
            if np.isfinite(lobe[:, k_max]).sum() > 20 else None,
            "bootstrap_replicates": 1000}


def h1_counts(C):
    return [int(np.sum(np.diff(h1_diagram(m), axis=1) > H1_TAU)) for m in C]


def topology(d, L, cR, rows, sel):
    g = d["g"]
    step = g[1] - g[0]
    mid = 0.5 * (g[1:] + g[:-1])
    R = L // 2 - 1
    Cs = d["C"][sel]
    entry = np.array([rows[i]["entry_crossing"] for i in sel], float)
    exit_ = np.array([rows[i]["exit_crossing"] for i in sel], float)
    dist = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    dist = np.minimum(dist, L - dist)
    shells = {w: np.argwhere(np.triu((np.abs(dist - R) <= w), 1)) for w in (0, 1)}
    velocity = {"H0_global": np.stack([landscape_velocity(np.stack([mst_deaths(c) for c in C]), step) for C in Cs])}
    for w, edges in shells.items():
        velocity[f"LR_landscape_w{w}"] = np.stack(
            [landscape_velocity(np.stack([shell_deaths(c, edges) for c in C]), step) for C in Cs])
    cRs = cR[sel]
    entry_mae, exit_mae = {}, {}
    for name, v in velocity.items():
        en, ex = split(v, cRs, g, mid, step)
        entry_mae[name], exit_mae[name] = mae(en, entry, step), mae(ex, exit_, step)
    fake = {"H0_velocity": velocity["H0_global"], "Gamma0_mid": mid, "Gamma0": g}
    entry_mae["H0_global_frozen_rule"] = mae(cm.topological_entry(fake), entry, step)
    exit_mae["H0_second_wave"] = mae(cm.second_wave(fake)[0], exit_, step)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        n1 = np.array(list(pool.map(h1_counts, list(Cs))))
    h1_exit = []
    for counts, e in zip(n1, entry):
        ok = (mid > e) & (np.diff(counts) >= 1)
        h1_exit.append(mid[np.flatnonzero(ok)[0]] if ok.any() else np.nan)
    exit_mae["H1_growth_tau0.04"] = mae(np.array(h1_exit), exit_, step)
    entry_mae["constant"] = mae(np.full_like(entry, np.median(entry)), entry, step)
    exit_mae["constant"] = mae(np.full_like(exit_, np.median(exit_)), exit_, step)
    shell0 = np.array([[1.0 - shell_deaths(c, shells[0]).mean() for c in C] for C in Cs])
    return {"selected": int(sel.size), "median_entry": float(np.median(entry)), "median_exit": float(np.median(exit_)),
            "entry_MAE": entry_mae, "exit_MAE": exit_mae,
            "w0_shell_edges": int(len(shells[0])), "w0_level_minus_C_R_max_abs": float(np.abs(shell0 - cRs).max())}


def obc_reference():
    out = {}
    if OBC_EVEN.exists():
        e = json.loads(OBC_EVEN.read_text(encoding="utf-8"))["L16"]
        out["OBC_L16"] = {"entry_MAE": e.get("C_entry_MAE"), "exit_MAE": e.get("C_exit_MAE")}
    if OBC_ODD.exists():
        o = json.loads(OBC_ODD.read_text(encoding="utf-8"))["L17"]
        out["OBC_L17"] = {"entry_MAE": o["C_entry_MAE"], "exit_MAE": o["C_exit_MAE"], "average_hump": o["E_average_hump"]}
    return out


def plot(results):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for L, color in ((16, "tab:blue"), (17, "tab:red")):
        p = results.get(f"L{L}", {}).get("P_physical")
        if p:
            lo, hi = np.array(p["band95"])
            axes[0].fill_between(RUN_GRID, lo, hi, color=color, alpha=0.2)
            axes[0].plot(RUN_GRID, p["mean_curve"], color=color, label=f"PBC L{L} ({p['seeds']} seeds)")
        m = results.get(f"L{L}", {}).get("M_h_map")
        if m:
            axes[1].plot(RUN_GRID, m["h_star_central"], "o-", ms=3, color=color, label=f"PBC L{L}")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set(xlabel=r"$\Gamma_0$", ylabel=r"$\overline{C}(R)$, $R=7$", title="h = 2.5, average over disorder")
    axes[1].set(xlabel=r"$\Gamma_0$", ylabel=r"$h^*_L(\Gamma_0)$", title="zero of the average (isotonic in h)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(OUT / "pbc_ED_parity_overview.png", dpi=200)
    fig.savefig(OUT / "pbc_ED_parity_overview.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true", help="QC on finished seeds; no topology, not for results")
    parser.add_argument("--main-only", action="store_true", help="complete main cohort only (h map still running)")
    args = parser.parse_args()
    if RUN_SELECTION != SELECTION:
        raise ValueError("runner selection differs from the frozen OBC selection")
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    main_c = spec["cohorts"]["main"]
    seeds = set(range(main_c["seed_range"][0], main_c["seed_range"][1] + 1))
    results = {"partial": args.partial, "spec": str(SPEC.relative_to(ROOT))}
    for L in main_c["sizes"]:
        res = {}
        d = load(L, main_c["h"][0], seeds, args.partial)
        if d is not None:
            res["P_physical"], cR, rows, sel = physical(d, L)
            if not args.partial and sel.size:
                res["T_topology"] = topology(d, L, cR, rows, sel)
        res["M_h_map"] = None if args.main_only else h_map(L, spec, args.partial)
        results[f"L{L}"] = res
        print(L, "done", flush=True)
    results["OBC_reference"] = obc_reference()
    name = "summary_partial.json" if args.partial else "summary_main.json" if args.main_only else "summary.json"
    (OUT / name).write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    plot(results)
    print(OUT / name)


if __name__ == "__main__":
    main()
