"""Validate the frozen H0 normalized-persistence-entropy curvature detector.

The detector was selected on OBC L=16 reentrant data and is transferred here
without tuning to clean h=0, random fixed_mean h=0, and the clean h!=0 L=64
frontier.  Input matrices are always C=Cstag; H0 finite bars are the MST edge
weights of D=1-C.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_clean_L64_new_topology import landscape, landscape_speed, mst_deaths  # noqa: E402
from analyze_clean_frontier_multisize import PAPER, save_figure  # noqa: E402
from analyze_random_h0_irfp_pseudocritical_scaling import local_quadratic_peak  # noqa: E402
from build_clean_frontier_H0_H1_L64_expanded import load_curves  # noqa: E402


OUTPUT = ROOT / "results/H0-entropy-curvature-cross-regime-validation-v1/analysis"
RANDOM_ROOT = ROOT / "results/random_ising_gamma_ensemble"
FINAL_RANDOM = RANDOM_ROOT / "random-h0-landscape-velocity-final-fine50-v1/analysis/random_h0_irfp_pseudocritical_scaling.npz"
RNG_SEED = 20261012
N_BOOT = 5000
EPS = 1e-14
USED_SOURCES: set[Path] = set()


def entropy_from_matrix(matrix: np.ndarray) -> float:
    deaths = mst_deaths(np.asarray(matrix, float))
    total = float(np.sum(deaths))
    positive = deaths[deaths > 0]
    if total <= EPS or deaths.size <= 1 or not positive.size:
        return 0.0
    p = positive / total
    return float(-np.sum(p * np.log(p)) / np.log(deaths.size))


def frozen_curvature(gamma: np.ndarray, curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return centres/response, excluding the first available centre.

    On uniform grids this is exactly -diff(curve, n=2)/step**2, as frozen on
    the reentrant cohorts.  The divided-difference form extends it to the few
    clean-frontier cuts whose merged grid is nonuniform.
    """
    gamma = np.asarray(gamma, float)
    curve = np.asarray(curve, float)
    left = np.diff(curve)[:-1] / np.diff(gamma)[:-1]
    right = np.diff(curve)[1:] / np.diff(gamma)[1:]
    response = -2.0 * (right - left) / (gamma[2:] - gamma[:-2])
    centres = gamma[1:-1]
    return centres[1:], response[1:]


def frozen_location(gamma: np.ndarray, curve: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    x, response = frozen_curvature(gamma, curve)
    return float(x[int(np.nanargmax(response))]), x, response


def clean_h0_sources() -> list[tuple[int, int, Path]]:
    choices = {
        8: (64, ROOT / "results/clean_ising_gamma_campaign/clean-h0-Gamma-critical-L8-64-chi64-v1/scans/L8_h_0_J1_chi64/continuation_summary.npz"),
        16: (64, ROOT / "results/clean_ising_gamma_campaign/clean-h0-Gamma-critical-L8-64-chi64-v1/scans/L16_h_0_J1_chi64/continuation_summary.npz"),
        32: (128, ROOT / "results/clean_ising_gamma_campaign/clean-h0-Gamma-critical-L32-64-chi128-v1/scans/L32_h_0_J1_chi128/continuation_summary.npz"),
        64: (128, ROOT / "results/clean_ising_gamma_campaign/clean-h0-Gamma-critical-L32-64-chi128-v1/scans/L64_h_0_J1_chi128/continuation_summary.npz"),
        128: (256, ROOT / "results/clean_ising_gamma_campaign/clean-h0-Gamma-critical-L128-chi256-v1/scans/L128_h_0_J1_chi256/continuation_summary.npz"),
    }
    return [(L, chi, path) for L, (chi, path) in choices.items()]


def analyse_clean_h0() -> tuple[list[dict], dict[str, np.ndarray]]:
    historical: dict[int, list[tuple[float, float]]] = {}
    velocity_csv = ROOT / "paper_figures/figure_clean_h0/output/clean_h0_landscape_velocity_by_L_g.csv"
    USED_SOURCES.add(velocity_csv)
    with velocity_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            historical.setdefault(int(row["L"]), []).append((float(row["g"]), float(row["xi_lambda"])))
    rows, arrays = [], {}
    for L, chi, path in clean_h0_sources():
        USED_SOURCES.add(path)
        with np.load(path, allow_pickle=False) as z:
            gamma = np.asarray(z["Gamma_values"], float)
            matrices = np.asarray(z["Cstag"], float)
            qc = np.asarray(z["branch_disagreement"], bool) | (np.asarray(z["selected_max_bond_dimension"], int) >= chi)
        entropy = np.asarray([entropy_from_matrix(m) for m in matrices])
        ent_peak, cx, curvature = frozen_location(gamma, entropy)
        values = sorted(historical[L])
        velocity_gamma = np.asarray([v[0] for v in values])
        velocity = np.asarray([v[1] for v in values])
        vel_peak = float(local_quadratic_peak(velocity_gamma, velocity)["Gamma0"])
        rows.append({"regime": "clean_h0", "L": L, "chi": chi, "points": gamma.size,
                     "entropy_curvature_Gamma": ent_peak, "landscape_velocity_Gamma": vel_peak,
                     "entropy_abs_error_exact": abs(ent_peak - 1.0), "velocity_abs_error_exact": abs(vel_peak - 1.0),
                     "branch_or_saturation_points": int(np.sum(qc)), "source": str(path.relative_to(ROOT).as_posix())})
        arrays.update({f"clean_L{L}_gamma": gamma, f"clean_L{L}_entropy": entropy,
                       f"clean_L{L}_curvature_gamma": cx, f"clean_L{L}_curvature": curvature,
                       f"clean_L{L}_velocity_gamma": velocity_gamma, f"clean_L{L}_velocity": velocity})
    return rows, arrays


def scan_seed(path: Path) -> int | None:
    match = re.search(r"_seed(\d+)_", path.parent.name)
    return int(match.group(1)) if match else None


def random_matrix_sources(L: int, target: np.ndarray, seeds: np.ndarray) -> dict[int, np.ndarray]:
    if L <= 32:
        chi = 64 if L <= 16 else 128
        campaign = RANDOM_ROOT / f"random-h0-Gamma-critical-L{L}-c{chi}-s0-49-v1/scans"
        paths = list(campaign.glob("*/continuation_summary.npz"))
    elif L == 64:
        paths = list((RANDOM_ROOT / "random-h0-Gamma-critical-L64-c128-s0-49-v1/scans").glob("*/continuation_summary.npz"))
        paths += list(RANDOM_ROOT.glob("random-h0-L64-fine625-c128-*/scans/*/continuation_summary.npz"))
    elif L == 128:
        paths = list(RANDOM_ROOT.glob("random-h0-Gamma-critical-L128-c256-*/scans/*/continuation_summary.npz"))
        paths += list(RANDOM_ROOT.glob("random-h0-L128-fine625-c256-*/scans/*/continuation_summary.npz"))
    else:
        raise ValueError(L)
    wanted = set(map(int, seeds))
    lookup: dict[int, dict[float, np.ndarray]] = {s: {} for s in wanted}
    for path in paths:
        seed = scan_seed(path)
        if seed not in wanted:
            continue
        USED_SOURCES.add(path)
        with np.load(path, allow_pickle=False) as z:
            gamma = np.asarray(z["Gamma0_values"], float)
            matrices = np.asarray(z["Cstag"], float)
        for g, matrix in zip(gamma, matrices, strict=True):
            key = round(float(g), 12)
            if any(abs(key - float(t)) < 1e-11 for t in target):
                old = lookup[seed].get(key)
                if old is not None and not np.allclose(old, matrix, atol=2e-7, rtol=2e-7):
                    # Independent reruns can differ slightly; retain the first and record no averaging.
                    continue
                lookup[seed][key] = matrix
    output = {}
    for seed in seeds:
        missing = [g for g in target if round(float(g), 12) not in lookup[int(seed)]]
        if missing:
            raise RuntimeError(f"L={L} seed={seed}: missing target points {missing}")
        output[int(seed)] = np.stack([lookup[int(seed)][round(float(g), 12)] for g in target])
    return output


def paired_stats(prediction: np.ndarray, physical: np.ndarray, velocity: np.ndarray, step: float, rng: np.random.Generator) -> dict:
    ae = np.abs(prediction - physical); ae_v = np.abs(velocity - physical); delta = ae - ae_v
    draws = rng.integers(0, prediction.size, size=(N_BOOT, prediction.size))
    rho = spearmanr(prediction, physical)
    return {"N": int(prediction.size), "step": float(step), "MAE_entropy": float(np.mean(ae)),
            "MAE_velocity": float(np.mean(ae_v)), "within_step_entropy": int(np.sum(ae <= step + 1e-12)),
            "within_step_velocity": int(np.sum(ae_v <= step + 1e-12)),
            "paired_delta_MAE_entropy_minus_velocity": float(np.mean(delta)),
            "paired_delta_ci95": [float(x) for x in np.quantile(np.mean(delta[draws], axis=1), [.025, .975])],
            "spearman_entropy_physical": float(rho.statistic), "spearman_p": float(rho.pvalue)}


def analyse_random_h0() -> tuple[list[dict], list[dict], dict, dict[str, np.ndarray]]:
    rng = np.random.default_rng(RNG_SEED)
    per_seed, summary, arrays = [], [], {}
    with np.load(FINAL_RANDOM, allow_pickle=False) as final:
        USED_SOURCES.add(FINAL_RANDOM)
        final_data = {k: np.asarray(final[k]) for k in final.files}
    for L in (8, 16, 32, 64, 128):
        gamma = final_data[f"L{L}_Gamma0"].astype(float)
        seeds = final_data[f"L{L}_seeds"].astype(int)
        physical = final_data[f"L{L}_physical_Gamma0_star_by_seed"].astype(float)
        velocity_peak = final_data[f"L{L}_landscape_Gamma0_star_by_seed"].astype(float)
        sources = random_matrix_sources(L, gamma, seeds)
        entropy = np.stack([[entropy_from_matrix(m) for m in sources[int(seed)]] for seed in seeds])
        prediction = np.asarray([frozen_location(gamma, row)[0] for row in entropy])
        step = float(np.min(np.diff(gamma)))
        stats = paired_stats(prediction, physical, velocity_peak, step, rng)
        ensemble_entropy = np.mean(entropy, axis=0)
        ensemble_peak, cx, response = frozen_location(gamma, ensemble_entropy)
        stats.update({"regime": "random_h0", "L": L, "ensemble_entropy_curvature_Gamma": ensemble_peak,
                      "ensemble_velocity_Gamma": float(final_data["ensemble_landscape_Gamma0_star"][np.flatnonzero(np.array([8,16,32,64,128]) == L)[0]]),
                      "ensemble_physical_Gamma": float(final_data["ensemble_physical_Gamma0_star"][np.flatnonzero(np.array([8,16,32,64,128]) == L)[0]])})
        summary.append(stats)
        for seed, p, e, v in zip(seeds, physical, prediction, velocity_peak, strict=True):
            per_seed.append({"L": L, "seed": int(seed), "physical_Gamma": p, "entropy_curvature_Gamma": e,
                             "landscape_velocity_Gamma": v, "AE_entropy": abs(e-p), "AE_velocity": abs(v-p)})
        arrays.update({f"random_L{L}_gamma": gamma, f"random_L{L}_seeds": seeds,
                       f"random_L{L}_entropy": entropy, f"random_L{L}_ensemble_curvature_gamma": cx,
                       f"random_L{L}_ensemble_curvature": response})
        print(f"random h=0 L={L}: {len(seeds)} seeds", flush=True)
    sizes = np.asarray([row["L"] for row in summary], float)
    mask = sizes >= 16
    fits = {}
    for field in ("ensemble_entropy_curvature_Gamma", "ensemble_velocity_Gamma", "ensemble_physical_Gamma"):
        values = np.asarray([row[field] for row in summary])
        slope, intercept = np.polyfit(1/sizes[mask], values[mask], 1)
        fits[field] = {"Gamma_infinity": float(intercept), "slope": float(slope),
                       "RMSE_to_exact": float(np.sqrt(np.mean((values[mask]-1)**2)))}
    return per_seed, summary, fits, arrays


def analyse_clean_frontier() -> tuple[list[dict], dict[str, np.ndarray]]:
    rows, arrays = [], {}
    curves, _ = load_curves()
    for curve in curves:
        USED_SOURCES.update(Path(p) for p in curve["sources"])
        h = float(curve["h"]); gamma = np.asarray(curve["Gamma"], float); matrices = np.asarray(curve["Cstag"], float)
        entropy = np.asarray([entropy_from_matrix(m) for m in matrices])
        ent_peak, cx, curvature = frozen_location(gamma, entropy)
        deaths = [mst_deaths(m) for m in matrices]
        landscapes = np.stack([landscape(np.column_stack((np.zeros(d.size), d))) for d in deaths])
        velocity = landscape_speed(landscapes, gamma); mid = (gamma[:-1]+gamma[1:])/2
        point_qc = np.asarray(curve["branch_disagreement"], bool) | (np.asarray(curve["bond"], int) >= 128)
        safe_curv = ~(point_qc[2:-1] | point_qc[1:-2] | point_qc[3:]) if point_qc.size >= 5 else np.zeros_like(curvature, bool)
        safe_ent_peak = float(cx[np.argmax(np.where(safe_curv, curvature, -np.inf))]) if np.any(safe_curv) else math.nan
        safe_vel = ~(point_qc[:-1] | point_qc[1:])
        safe_vel_peak = float(mid[np.argmax(np.where(safe_vel, velocity, -np.inf))]) if np.any(safe_vel) else math.nan
        reference = float(np.interp(h, sorted(PAPER), [PAPER[x] for x in sorted(PAPER)]))
        rows.append({"regime": "clean_frontier_L64", "h": h, "points": gamma.size, "paper_Gamma": reference,
                     "entropy_curvature_Gamma_all": ent_peak, "entropy_curvature_Gamma_qc_safe": safe_ent_peak,
                     "landscape_velocity_Gamma_qc_safe_discrete": safe_vel_peak,
                     "AE_entropy_all": abs(ent_peak-reference), "AE_entropy_qc_safe": abs(safe_ent_peak-reference),
                     "AE_velocity_qc_safe": abs(safe_vel_peak-reference), "QC_flag": bool(np.any(point_qc)),
                     "QC_point_count": int(np.sum(point_qc))})
        tag = str(h).replace('.', 'p')
        arrays.update({f"frontier_h{tag}_gamma": gamma, f"frontier_h{tag}_entropy": entropy,
                       f"frontier_h{tag}_curvature_gamma": cx, f"frontier_h{tag}_curvature": curvature})
    return rows, arrays


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def plot(clean_rows: list[dict], random_summary: list[dict], frontier: list[dict]) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), constrained_layout=True)
    L = np.asarray([r["L"] for r in clean_rows]); x = 1/L
    axes[0].plot(x, [r["entropy_curvature_Gamma"] for r in clean_rows], "o-", label="entropy curvature")
    axes[0].plot(x, [r["landscape_velocity_Gamma"] for r in clean_rows], "s--", label="landscape velocity")
    axes[0].axhline(1, color="k", ls=":", label="exact")
    axes[0].set(xlabel=r"$1/L$", ylabel=r"$\Gamma^\star$", title=r"Clean $h=0$")
    sizes = np.asarray([r["L"] for r in random_summary]); xx = 1/sizes
    axes[1].plot(xx, [r["ensemble_entropy_curvature_Gamma"] for r in random_summary], "o-", label="entropy curvature")
    axes[1].plot(xx, [r["ensemble_velocity_Gamma"] for r in random_summary], "s--", label="landscape velocity")
    axes[1].plot(xx, [r["ensemble_physical_Gamma"] for r in random_summary], "^-.", label="physical")
    axes[1].axhline(1, color="k", ls=":")
    axes[1].set(xlabel=r"$1/L$", ylabel=r"$\Gamma_0^\star$", title=r"Random $h=0$ (ensemble)")
    ph = np.asarray(sorted(PAPER)); axes[2].plot(ph, [PAPER[v] for v in ph], "k-", label="digitized frontier")
    h = np.asarray([r["h"] for r in frontier]); axes[2].plot(h, [r["entropy_curvature_Gamma_qc_safe"] for r in frontier], "o-", label="entropy curvature")
    axes[2].plot(h, [r["landscape_velocity_Gamma_qc_safe_discrete"] for r in frontier], "s--", label="landscape velocity")
    axes[2].set(xlabel=r"$h$", ylabel=r"$\Gamma^\star$", title=r"Clean frontier $L=64$")
    for i, ax in enumerate(axes):
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8); ax.text(.02,.96,f"({chr(97+i)})",transform=ax.transAxes,va="top",fontweight="bold")
    save_figure(fig, OUTPUT / "H0_entropy_curvature_cross_regime_validation.png", dpi=260)
    save_figure(fig, OUTPUT / "H0_entropy_curvature_cross_regime_validation.pdf")
    plt.close(fig)


def main() -> None:
    matplotlib.use("Agg")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    clean, a1 = analyse_clean_h0()
    per_seed, random_summary, fits, a2 = analyse_random_h0()
    frontier, a3 = analyse_clean_frontier()
    write_csv(OUTPUT / "clean_h0_comparison.csv", clean)
    write_csv(OUTPUT / "random_h0_per_seed.csv", per_seed)
    write_csv(OUTPUT / "random_h0_summary.csv", random_summary)
    write_csv(OUTPUT / "clean_frontier_L64_comparison.csv", frontier)
    np.savez_compressed(OUTPUT / "validation_curves.npz", **a1, **a2, **a3)
    manifest_rows = [{"path": str(path.resolve().relative_to(ROOT.resolve()).as_posix()),
                      "size_bytes": path.stat().st_size} for path in sorted(USED_SOURCES)]
    write_csv(OUTPUT / "input_manifest.csv", manifest_rows)
    clean_fit = {}
    for field in ("entropy_curvature_Gamma", "landscape_velocity_Gamma"):
        L = np.asarray([r["L"] for r in clean], float); y = np.asarray([r[field] for r in clean]); mask=L>=16
        slope, intercept=np.polyfit(1/L[mask],y[mask],1)
        clean_fit[field]={"Gamma_infinity":float(intercept),"slope":float(slope),"RMSE_to_exact":float(np.sqrt(np.mean((y[mask]-1)**2)))}
    valid = [r for r in frontier if np.isfinite(r["entropy_curvature_Gamma_qc_safe"])]
    frontier_metrics = {"N":len(frontier),"N_qc_safe_entropy":len(valid),
                        "MAE_entropy_all":float(np.mean([r["AE_entropy_all"] for r in frontier])),
                        "MAE_entropy_qc_safe":float(np.mean([r["AE_entropy_qc_safe"] for r in valid])),
                        "MAE_velocity_qc_safe":float(np.mean([r["AE_velocity_qc_safe"] for r in frontier if np.isfinite(r["AE_velocity_qc_safe"])]))}
    summary={"created_at_utc":datetime.now(timezone.utc).isoformat(),"detector":"maximum of negative second derivative of normalized H0 persistent entropy; first curvature centre excluded; no tuning",
             "conventions":{"bc":"OBC","random_ensemble":"fixed_mean","matrix":"C=Cstag","distance":"D=1-C"},
             "clean_h0_fit_L_ge_16":clean_fit,"random_h0_fit_L_ge_16":fits,"random_h0_by_size":random_summary,
             "clean_frontier_L64":frontier_metrics,
             "guardrails":["Transferred detector frozen on reentrant L16; no orientation, smoothing, or extremum selected here.","Clean-frontier digitized curve is a thermodynamic reference; L64 points are finite-size OBC estimates.","QC flags retained; all-point and QC-safe entropy results are both reported."]}
    (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    plot(clean,random_summary,frontier)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
