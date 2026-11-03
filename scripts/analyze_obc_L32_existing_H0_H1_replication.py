"""Replicate the key OBC L16 H0/H1 analyses on all existing L32 data.

The physical selection is performed before topology.  The four fields with
three independent 50-seed cohorts are analysed separately; repeated seeds
across fields are never pooled as independent observations.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_clean_L64_new_topology import h1_diagram, mst_deaths  # noqa: E402
from analyze_pbc_reentrant_selected_seed_topology import (  # noqa: E402
    linear_zero_crossings,
    select_physical_seeds,
)


BASE = ROOT / "results" / "random_ising_gamma_ensemble"
OUTPUT = BASE / "L32-existing-H0-H1-replication-v1" / "analysis"
FIELDS = np.asarray([2.3000, 2.3125, 2.3250, 2.3375])
THRESHOLDS = np.asarray(
    tuple(np.round(np.arange(0.001, 0.0201, 0.001), 3))
    + (0.025, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100)
)
GAMMA = np.arange(13, dtype=float) * 0.05
MID = 0.5 * (GAMMA[:-1] + GAMMA[1:])
GRID = np.linspace(0.0, 2.0, 401)
LAYERS = 31
N_BOOT = 2000
RNG_SEED = 20261009
SELECTION = {
    "endpoint_points": 2,
    "negative_margin": 0.02,
    "positive_amplitude": 0.05,
    "minimum_lobe_width": 0.10,
    "minimum_positive_run": 2,
}
CLASS_NAMES = ("reentrant", "single_exit", "always_negative")


def landscape_from_deaths(deaths: np.ndarray) -> np.ndarray:
    diagram = np.column_stack((np.zeros(deaths.size), deaths))
    tents = np.maximum(
        np.minimum(GRID[None, :] - diagram[:, 0, None], diagram[:, 1, None] - GRID[None, :]),
        0.0,
    )
    ranked = np.sort(tents, axis=0)[::-1]
    result = np.zeros((LAYERS, GRID.size), np.float32)
    count = min(LAYERS, ranked.shape[0])
    result[:count] = ranked[:count]
    return result


def topology_one(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h0 = landscape_from_deaths(mst_deaths(matrix)[:LAYERS])
    diagram = h1_diagram(matrix)
    lifetime = diagram[:, 1] - diagram[:, 0] if diagram.size else np.empty(0)
    h1_count = np.sum(lifetime[:, None] > THRESHOLDS[None, :], axis=0).astype(np.uint16)
    total = float(np.sum(lifetime))
    effective = float(total * total / np.sum(lifetime**2)) if lifetime.size and np.sum(lifetime**2) > 0 else 0.0
    h1_scalar = np.asarray([
        float(np.max(lifetime)) if lifetime.size else 0.0,
        total,
        float(lifetime.size),
        effective,
    ])
    return h0, h1_count, h1_scalar


def load_existing() -> dict[str, np.ndarray]:
    matrices = np.empty((FIELDS.size, 150, GAMMA.size, 32, 32), np.float64)
    branch = np.zeros((FIELDS.size, 150, GAMMA.size), bool)
    saturated = np.zeros_like(branch)

    for seed_start, campaign in (
        (0, "L32-reentrant-map-expanded-c64-s0-49-v1"),
        (50, "L32-reentrant-confirm-c64-s50-99-v1"),
    ):
        with np.load(BASE / campaign / "analysis" / "ensemble_summary.npz", allow_pickle=False) as archive:
            h_grid = np.asarray(archive["h_values"], float)
            gamma = np.asarray(archive["Gamma0_values"], float)
            if not np.allclose(gamma, GAMMA):
                raise ValueError(f"Gamma grid mismatch in {campaign}")
            h_indices = [int(np.flatnonzero(np.isclose(h_grid, h))[0]) for h in FIELDS]
            matrices[:, seed_start:seed_start + 50] = np.asarray(archive["Cstag_by_seed"], float)[h_indices]
            branch[:, seed_start:seed_start + 50] = np.asarray(archive["branch_disagreement_by_seed"], bool)[h_indices]
            saturated[:, seed_start:seed_start + 50] = np.asarray(archive["bond_saturation_by_seed"], bool)[h_indices]

    found: set[tuple[int, int]] = set()
    campaign_paths = sorted(BASE.glob("L32-reentrant-efficiency-third*"))
    for campaign in campaign_paths:
        for path in sorted((campaign / "scans").glob("*/continuation_summary.npz")):
            match = re.search(r"_seed(\d+)_", path.parent.name)
            if match is None:
                continue
            seed = int(match.group(1))
            if not 100 <= seed <= 149:
                continue
            with np.load(path, allow_pickle=False) as archive:
                h = float(archive["h"])
                candidates = np.flatnonzero(np.isclose(FIELDS, h))
                if candidates.size != 1:
                    continue
                hi = int(candidates[0])
                if (hi, seed) in found:
                    raise ValueError(f"duplicate third-cohort scan h={h}, seed={seed}")
                if not np.allclose(archive["Gamma0_values"], GAMMA):
                    raise ValueError(f"Gamma mismatch: {path}")
                matrices[hi, seed] = np.asarray(archive["Cstag"], float)
                branch[hi, seed] = np.asarray(archive["branch_disagreement"], bool)
                saturated[hi, seed] = np.maximum(
                    np.asarray(archive["forward_max_bond_dimension"], int),
                    np.asarray(archive["backward_max_bond_dimension"], int),
                ) >= 64
                found.add((hi, seed))
    if len(found) != 200:
        raise ValueError(f"third cohort incomplete: {len(found)}/200 scans")

    physical = np.diagonal(matrices, offset=15, axis1=3, axis2=4).mean(axis=3)
    return {
        "matrices": matrices,
        "physical": physical,
        "branch": branch,
        "saturated": saturated,
        "seeds": np.arange(150, dtype=int),
    }


def physical_classes(physical: np.ndarray, seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    class_code = np.full((FIELDS.size, seeds.size), "other", dtype="U20")
    entry = np.full((FIELDS.size, seeds.size), np.nan)
    exit_ = np.full_like(entry, np.nan)
    rows_out: list[dict[str, object]] = []
    for hi, h in enumerate(FIELDS):
        selection_rows, selected = select_physical_seeds(GAMMA, seeds, physical[hi], **SELECTION)
        selected_set = set(selected.tolist())
        for si, (seed, curve, selection_row) in enumerate(zip(seeds, physical[hi], selection_rows, strict=True)):
            crossings = linear_zero_crossings(GAMMA, curve)
            low, high = float(np.mean(curve[:2])), float(np.mean(curve[-2:]))
            if si in selected_set:
                code = "reentrant"
                entry[hi, si], exit_[hi, si] = crossings
            elif len(crossings) == 1 and low > 0.0 and high < 0.0:
                code = "single_exit"
                exit_[hi, si] = crossings[0]
            elif np.all(curve < 0.0):
                code = "always_negative"
            elif np.all(curve > 0.0):
                code = "always_positive"
            else:
                code = "other"
            class_code[hi, si] = code
            rows_out.append({
                "h": float(h), "seed": int(seed), "physical_class": code,
                "entry": float(entry[hi, si]) if np.isfinite(entry[hi, si]) else "",
                "exit": float(exit_[hi, si]) if np.isfinite(exit_[hi, si]) else "",
                **selection_row,
            })
    return class_code, entry, exit_, rows_out


def h0_velocity(landscapes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    derivative = np.diff(landscapes.astype(float), axis=1) / np.diff(GAMMA)[None, :, None, None]
    individual = np.sqrt(np.trapezoid(np.sum(derivative**2, axis=2), GRID, axis=2) / LAYERS)
    return individual, derivative


def first_half(active_fraction: np.ndarray) -> float:
    indices = np.flatnonzero(active_fraction >= 0.5)
    return float(GAMMA[indices[0]]) if indices.size else float("nan")


def ci(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def bootstrap_h0(velocity: np.ndarray, derivative: np.ndarray, rng: np.random.Generator) -> dict[str, object]:
    n = velocity.shape[0]
    draws = rng.integers(0, n, size=(N_BOOT, n))
    mean_boot = np.mean(velocity[draws], axis=1)
    mean_boot[:, 0] = -np.inf
    mean_peaks = MID[np.argmax(mean_boot, axis=1)]
    mean_curve = np.mean(velocity, axis=0)
    mean_curve[0] = np.nan

    weights_t = np.full(GRID.size, float(GRID[1] - GRID[0]))
    weights_t[[0, -1]] *= 0.5
    flat = (derivative * np.sqrt(weights_t[None, None, None, :] / LAYERS)).reshape(n, MID.size, -1)
    gram = np.einsum("igk,jgk->gij", flat, flat, optimize=True)
    weights = np.zeros((N_BOOT, n), np.float32)
    np.add.at(weights, (np.arange(N_BOOT)[:, None], draws), 1.0 / n)
    collective_boot = np.sqrt(np.maximum(np.einsum("bi,gij,bj->bg", weights, gram, weights, optimize=True), 0.0))
    collective_boot[:, 0] = -np.inf
    collective_peaks = MID[np.argmax(collective_boot, axis=1)]
    equal = np.full(n, 1.0 / n)
    collective_curve = np.sqrt(np.maximum(np.einsum("i,gij,j->g", equal, gram, equal, optimize=True), 0.0))
    collective_curve[0] = np.nan
    return {
        "mean_curve": mean_curve,
        "collective_curve": collective_curve,
        "mean_peak": float(MID[np.nanargmax(mean_curve)]),
        "collective_peak": float(MID[np.nanargmax(collective_curve)]),
        "mean_ci": ci(mean_peaks),
        "collective_ci": ci(collective_peaks),
    }


def bootstrap_half_active(active: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = active.shape[0]
    draws = rng.integers(0, n, size=(N_BOOT, n))
    boot = np.mean(active[draws], axis=1)
    locations = np.asarray([first_half(curve) for curve in boot])
    return ci(locations)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    loaded = load_existing()
    matrices = loaded["matrices"]
    physical = loaded["physical"]
    seeds = loaded["seeds"]
    class_code, entry, exit_, physical_rows = physical_classes(physical, seeds)
    write_csv(OUTPUT / "physical_classification.csv", physical_rows)

    h0_all = np.empty((FIELDS.size, seeds.size, MID.size), np.float32)
    h1_count = np.empty((FIELDS.size, seeds.size, GAMMA.size, THRESHOLDS.size), np.uint16)
    h1_scalar = np.empty((FIELDS.size, seeds.size, GAMMA.size, 4), np.float32)
    h0_aggregate: dict[str, dict[str, object]] = {}
    rng = np.random.default_rng(RNG_SEED)
    h0_rows: list[dict[str, object]] = []

    for hi, h in enumerate(FIELDS):
        flat_matrices = matrices[hi].reshape(-1, 32, 32)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(topology_one, flat_matrices, chunksize=8))
        landscapes = np.stack([result[0] for result in results]).reshape(seeds.size, GAMMA.size, LAYERS, GRID.size)
        h1_count[hi] = np.stack([result[1] for result in results]).reshape(seeds.size, GAMMA.size, THRESHOLDS.size)
        h1_scalar[hi] = np.stack([result[2] for result in results]).reshape(seeds.size, GAMMA.size, 4)
        velocity, derivative = h0_velocity(landscapes)
        h0_all[hi] = velocity.astype(np.float32)
        mask = class_code[hi] == "reentrant"
        aggregate = bootstrap_h0(velocity[mask], derivative[mask], rng)
        h0_aggregate[f"{h:.4f}"] = aggregate
        peak_index = 1 + np.argmax(velocity[mask, 1:], axis=1)
        individual_peak = MID[peak_index]
        errors = individual_peak - entry[hi, mask]
        h0_rows.append({
            "h": float(h), "reentrant_count": int(mask.sum()),
            "entry_median": float(np.median(entry[hi, mask])),
            "individual_match_within_one_grid_step": int(np.sum(np.abs(errors) <= 0.05 + 1e-12)),
            "individual_MAE": float(np.mean(np.abs(errors))),
            "individual_median_AE": float(np.median(np.abs(errors))),
            "mean_of_velocities_peak": aggregate["mean_peak"],
            "mean_of_velocities_ci_low": aggregate["mean_ci"][0],
            "mean_of_velocities_ci_high": aggregate["mean_ci"][1],
            "velocity_of_mean_peak": aggregate["collective_peak"],
            "velocity_of_mean_ci_low": aggregate["collective_ci"][0],
            "velocity_of_mean_ci_high": aggregate["collective_ci"][1],
        })
        print(f"topology complete for h={h:.4f}: {flat_matrices.shape[0]} matrices", flush=True)

    write_csv(OUTPUT / "H0_reentrant_replication.csv", h0_rows)
    h1_rows: list[dict[str, object]] = []
    for hi, h in enumerate(FIELDS):
        for ti, threshold in enumerate(THRESHOLDS):
            for class_number, name in enumerate(CLASS_NAMES):
                mask = class_code[hi] == name
                values = h1_count[hi, mask, :, ti]
                active = values > 0
                fraction = np.mean(active, axis=0)
                half = first_half(fraction)
                low, high = bootstrap_half_active(active, rng)
                median_count = np.median(values, axis=0)
                growth = np.maximum(np.diff(median_count) / np.diff(GAMMA), 0.0)
                growth[0] = np.nan
                peak = float(MID[np.nanargmax(growth)]) if np.nanmax(growth) > 0 else float("nan")
                physical_reference = float(np.median(exit_[hi, mask])) if np.any(np.isfinite(exit_[hi, mask])) else float("nan")
                individual_onset = np.full(mask.sum(), np.nan)
                for index, curve in enumerate(active):
                    detected = np.flatnonzero(curve)
                    if detected.size:
                        individual_onset[index] = GAMMA[detected[0]]
                finite_pair = np.isfinite(individual_onset) & np.isfinite(exit_[hi, mask])
                paired_mae = float(np.mean(np.abs(individual_onset[finite_pair] - exit_[hi, mask][finite_pair]))) if np.any(finite_pair) else float("nan")
                h1_rows.append({
                    "h": float(h), "physical_class": name, "n_seeds": int(mask.sum()),
                    "threshold": float(threshold), "half_active_Gamma0": half,
                    "half_active_ci_low": low, "half_active_ci_high": high,
                    "median_growth_peak_Gamma0": peak,
                    "physical_exit_median": physical_reference,
                    "H1_minus_physical_exit": half - physical_reference if np.isfinite(physical_reference) else "",
                    "detected_individual_onsets": int(np.sum(np.isfinite(individual_onset))),
                    "paired_individual_onset_exit_MAE": paired_mae,
                    "active_fraction_at_0p6": float(fraction[-1]),
                    "mean_N1_at_0p6": float(np.mean(values[:, -1])),
                })
    write_csv(OUTPUT / "H1_physical_class_replication.csv", h1_rows)

    np.savez_compressed(
        OUTPUT / "analysis_arrays.npz", h_values=FIELDS, Gamma0=GAMMA, Gamma0_mid=MID,
        seeds=seeds, physical_curves=physical, physical_class=class_code,
        physical_entry=entry, physical_exit=exit_, branch_disagreement=loaded["branch"],
        chi_saturated=loaded["saturated"], H0_velocity=h0_all,
        H1_cycle_count=h1_count, H1_scalar=h1_scalar,
        H1_scalar_names=np.asarray(["maximum_lifetime", "total_persistence", "cycle_count", "effective_cycle_count"]),
        thresholds=THRESHOLDS,
    )

    plot_results(physical, class_code, entry, exit_, h0_all, h0_aggregate, h1_count, h1_scalar, h1_rows)

    class_counts = {
        f"{h:.4f}": {name: int(np.sum(class_code[hi] == name)) for name in np.unique(class_code[hi])}
        for hi, h in enumerate(FIELDS)
    }
    summary = {
        "status": "complete L32 existing-data replication",
        "protocol": {"L": 32, "bc": "open", "chi": 64, "ensemble": "fixed_mean", "matrix": "C=Cstag", "D": "1-C internally"},
        "coverage": {"fields": FIELDS.tolist(), "seeds_per_field": 150, "Gamma0": GAMMA.tolist(), "matrix_count": int(FIELDS.size * seeds.size * GAMMA.size)},
        "physical_selection": SELECTION,
        "class_counts": class_counts,
        "QC": {
            "branch_disagreement_points": int(np.sum(loaded["branch"])),
            "branch_disagreement_seed_field_pairs": int(np.sum(np.any(loaded["branch"], axis=2))),
            "chi_saturated_points": int(np.sum(loaded["saturated"])),
            "chi_saturated_seed_field_pairs": int(np.sum(np.any(loaded["saturated"], axis=2))),
            "policy": "flags preserved; no exclusions in primary analysis",
        },
        "H0": h0_rows,
        "guardrails": [
            "The four h cuts are analysed separately because the same seeds recur across fields.",
            "The L16 absolute H1 thresholds are transferred without tuning.",
            "The Gamma0 grid is four times coarser than the L16 grid (0.05 versus 0.0125).",
            "Physical selection is topology blind and uses the frozen L16 rule, but coarse-grid sensitivity must be reported.",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def plot_results(
    physical: np.ndarray,
    class_code: np.ndarray,
    entry: np.ndarray,
    exit_: np.ndarray,
    h0_velocity_values: np.ndarray,
    h0_aggregate: dict[str, dict[str, object]],
    h1_count: np.ndarray,
    h1_scalar: np.ndarray,
    h1_rows: list[dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"reentrant": "#0072B2", "single_exit": "#D55E00", "always_negative": "#009E73"}

    figure, axes = plt.subplots(2, 2, figsize=(12.3, 8.8), constrained_layout=True)
    for hi, (h, axis) in enumerate(zip(FIELDS, axes.ravel(), strict=True)):
        for name in CLASS_NAMES:
            mask = class_code[hi] == name
            axis.plot(GAMMA, np.mean(physical[hi, mask], axis=0), color=colors[name], lw=2.0, label=f"{name} ({mask.sum()})")
            if name == "reentrant":
                for curve in physical[hi, mask]:
                    axis.plot(GAMMA, curve, color=colors[name], lw=0.55, alpha=0.18)
        axis.axhline(0, color="0.3", lw=0.8)
        axis.set(title=fr"$h={h:.4f}$", xlabel=r"$\Gamma_0$", ylabel="signed paper-distance Cstag", xlim=(0, .6))
        axis.grid(alpha=.17); axis.legend(frameon=False, fontsize=8)
    figure.suptitle("L=32 physical classes (150 seeds per field; no topology used)", fontsize=14)
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT / f"L32_physical_classes.{extension}", dpi=260)
    plt.close(figure)

    # Mechanistic H1 summaries at the field with the largest reentrant cohort.
    hi = int(np.argmax([np.sum(class_code[index] == "reentrant") for index in range(FIELDS.size)]))
    figure, axes = plt.subplots(2, 2, figsize=(12.3, 8.8), constrained_layout=True)
    labels = ("maximum lifetime", "total persistence", "raw cycle count", "effective cycle count")
    for scalar_index, (axis, label) in enumerate(zip(axes.ravel(), labels, strict=True)):
        for name in CLASS_NAMES:
            mask = class_code[hi] == name
            values = h1_scalar[hi, mask, :, scalar_index]
            mean = np.mean(values, axis=0)
            sem = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
            axis.plot(GAMMA, mean, color=colors[name], lw=2.0, label=name)
            axis.fill_between(GAMMA, mean-sem, mean+sem, color=colors[name], alpha=.13)
        axis.set(title=label, xlabel=r"$\Gamma_0$", xlim=(0,.6))
        axis.grid(alpha=.17); axis.legend(frameon=False, fontsize=8)
    figure.suptitle(fr"L=32 H1 persistence geometry at $h={FIELDS[hi]:.4f}$", fontsize=14)
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT / f"L32_H1_persistence_geometry.{extension}", dpi=260)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12.3, 8.8), constrained_layout=True)
    for hi, (h, axis) in enumerate(zip(FIELDS, axes.ravel(), strict=True)):
        mask = class_code[hi] == "reentrant"
        peak_index = 1 + np.argmax(h0_velocity_values[hi, mask, 1:], axis=1)
        peaks = MID[peak_index]
        axis.scatter(entry[hi, mask], peaks, s=38, color="#0072B2", alpha=.8)
        axis.plot([0, .35], [0, .35], color="0.35", ls="--")
        axis.set(title=fr"$h={h:.4f}$, N={mask.sum()}$", xlabel="physical entry", ylabel="individual H0 peak", xlim=(0, .35), ylim=(0, .35))
        axis.grid(alpha=.17)
    figure.suptitle("L=32: seed-resolved H0 entry replication", fontsize=14)
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT / f"L32_H0_entry_pairing.{extension}", dpi=260)
    plt.close(figure)

    ti = int(np.flatnonzero(np.isclose(THRESHOLDS, 0.010))[0])
    figure, axes = plt.subplots(2, 2, figsize=(12.3, 8.8), constrained_layout=True)
    for hi, (h, axis) in enumerate(zip(FIELDS, axes.ravel(), strict=True)):
        for name in CLASS_NAMES:
            mask = class_code[hi] == name
            fraction = np.mean(h1_count[hi, mask, :, ti] > 0, axis=0)
            axis.plot(GAMMA, fraction, color=colors[name], lw=2.1, label=f"{name} ({mask.sum()})")
        for name, style in (("reentrant", "--"), ("single_exit", ":")):
            mask = class_code[hi] == name
            axis.axvline(np.nanmedian(exit_[hi, mask]), color=colors[name], ls=style, lw=1.25)
        axis.axhline(.5, color="0.55", ls=":", lw=1)
        axis.set(title=fr"$h={h:.4f}$", xlabel=r"$\Gamma_0$", ylabel=r"fraction with $N_1>0$", xlim=(0,.6), ylim=(-.02,1.02))
        axis.grid(alpha=.17); axis.legend(frameon=False, fontsize=7.8)
    figure.suptitle(r"L=32 H1 activation at transferred threshold $\tau=0.010$", fontsize=14)
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT / f"L32_H1_three_classes_tau010.{extension}", dpi=260)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12.3, 8.8), constrained_layout=True)
    for hi, (h, axis) in enumerate(zip(FIELDS, axes.ravel(), strict=True)):
        for name in CLASS_NAMES:
            selected = [row for row in h1_rows if np.isclose(float(row["h"]), h) and row["physical_class"] == name]
            y = np.asarray([float(row["half_active_Gamma0"]) for row in selected])
            low = np.asarray([float(row["half_active_ci_low"]) for row in selected])
            high = np.asarray([float(row["half_active_ci_high"]) for row in selected])
            axis.errorbar(THRESHOLDS, y, yerr=np.vstack((y-low, high-y)), marker="o", color=colors[name], lw=1.8, capsize=2.5, label=name)
        axis.set(title=fr"$h={h:.4f}$", xlabel=r"threshold $\tau$", ylabel=r"first $\Gamma_0$ with active fraction $\geq1/2$", ylim=(0.2,.62))
        axis.grid(alpha=.17); axis.legend(frameon=False, fontsize=8)
    figure.suptitle("L=32 H1 population scale across physical classes", fontsize=14)
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT / f"L32_H1_threshold_class_comparison.{extension}", dpi=260)
    plt.close(figure)


if __name__ == "__main__":
    main()
