"""Topological response of PBC L=16 seeds with a clear physical reentrant lobe.

The seed selection is deliberately topology-blind.  At a fixed h, it uses the
signed paper-distance Cstag curve and requires a negative-positive-negative
trajectory with exactly two well-separated zero crossings.  Once that cohort
is frozen, the script reads the already computed H0/H1 persistence summaries
and Gamma-landscape Gram matrices and reports both individual curves and the
selected-cohort aggregate.

The selected-cohort landscape velocity is the norm of the derivative of the
mean landscape, as in the random h=0 analysis.  It is not the arithmetic mean
of the individual landscape velocities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = (
    ROOT
    / "results"
    / "random_ising_gamma_ensemble"
    / "pbc-reentrant-L16-final-fine625-c64-s10-59-view-v1"
)
SCALAR_NAMES = ("maximum_lifetime", "total_persistence", "persistent_entropy", "death_range")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, nargs="?", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--h", type=float, default=2.5)
    parser.add_argument("--endpoint-points", type=int, default=2)
    parser.add_argument("--negative-margin", type=float, default=0.02)
    parser.add_argument("--positive-amplitude", type=float, default=0.05)
    parser.add_argument("--minimum-lobe-width", type=float, default=0.10)
    parser.add_argument("--minimum-positive-run", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def linear_zero_crossings(x: np.ndarray, y: np.ndarray) -> list[float]:
    crossings: list[float] = []
    for index in range(x.size - 1):
        y0, y1 = float(y[index]), float(y[index + 1])
        if not np.isfinite(y0) or not np.isfinite(y1):
            continue
        if y0 == 0.0:
            crossings.append(float(x[index]))
        elif y0 * y1 < 0.0:
            crossings.append(
                float(x[index] - y0 * (x[index + 1] - x[index]) / (y1 - y0))
            )
    if y.size and float(y[-1]) == 0.0:
        crossings.append(float(x[-1]))
    unique: list[float] = []
    for value in crossings:
        if not unique or not np.isclose(value, unique[-1], atol=1.0e-12, rtol=0.0):
            unique.append(value)
    return unique


def longest_positive_run(values: np.ndarray) -> int:
    best = current = 0
    for positive in np.asarray(values) > 0.0:
        current = current + 1 if positive else 0
        best = max(best, current)
    return int(best)


def select_physical_seeds(
    gamma: np.ndarray,
    seeds: np.ndarray,
    curves: np.ndarray,
    *,
    endpoint_points: int,
    negative_margin: float,
    positive_amplitude: float,
    minimum_lobe_width: float,
    minimum_positive_run: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    rows: list[dict[str, object]] = []
    selected_indices: list[int] = []
    for seed_index, (seed, curve) in enumerate(zip(seeds, curves, strict=True)):
        crossings = linear_zero_crossings(gamma, curve)
        left_mean = float(np.mean(curve[:endpoint_points]))
        right_mean = float(np.mean(curve[-endpoint_points:]))
        maximum = float(np.max(curve))
        positive_run = longest_positive_run(curve)
        exactly_two = len(crossings) == 2
        width = float(crossings[1] - crossings[0]) if exactly_two else math.nan
        tests = {
            "finite_curve": bool(np.all(np.isfinite(curve))),
            "exactly_two_crossings": exactly_two,
            "negative_low_field": left_mean <= -negative_margin,
            "positive_inside": maximum >= positive_amplitude,
            "negative_high_field": right_mean <= -negative_margin,
            "minimum_lobe_width": exactly_two and width >= minimum_lobe_width,
            "minimum_positive_run": positive_run >= minimum_positive_run,
        }
        selected = all(tests.values())
        if selected:
            selected_indices.append(seed_index)
        failed = [name for name, passed in tests.items() if not passed]
        rows.append(
            {
                "seed": int(seed),
                "selected": selected,
                "entry_crossing": crossings[0] if exactly_two else math.nan,
                "exit_crossing": crossings[1] if exactly_two else math.nan,
                "crossing_count": len(crossings),
                "lobe_width": width,
                "low_field_mean": left_mean,
                "high_field_mean": right_mean,
                "maximum_signed_Cstag": maximum,
                "positive_run_points": positive_run,
                "failed_criteria": ";".join(failed),
            }
        )
    return rows, np.asarray(selected_indices, dtype=int)


def speed_from_gram(gram: np.ndarray, selected: np.ndarray) -> np.ndarray:
    weights = np.zeros(gram.shape[-1], dtype=float)
    weights[selected] = 1.0 / selected.size
    squared = np.einsum("s,gsr,r->g", weights, gram, weights, optimize=True)
    return np.sqrt(np.maximum(squared, 0.0))


def mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    mean = np.nanmean(array, axis=0)
    valid = np.sum(np.isfinite(array), axis=0)
    sem = np.full(mean.shape, np.nan, dtype=float)
    mask = valid > 1
    sem[mask] = np.nanstd(array[:, mask], axis=0, ddof=1) / np.sqrt(valid[mask])
    return mean, sem


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_boundary_guides(axis, entry: np.ndarray, exit_: np.ndarray) -> None:
    entry_median = float(np.median(entry))
    exit_median = float(np.median(exit_))
    axis.axvspan(float(np.min(entry)), float(np.max(entry)), color="#0072B2", alpha=0.07)
    axis.axvspan(float(np.min(exit_)), float(np.max(exit_)), color="#D55E00", alpha=0.07)
    axis.axvline(entry_median, color="#0072B2", linestyle="--", linewidth=1.0)
    axis.axvline(exit_median, color="#D55E00", linestyle="--", linewidth=1.0)


def plot_physical_selection(
    output: Path,
    gamma: np.ndarray,
    seeds: np.ndarray,
    curves: np.ndarray,
    selected: np.ndarray,
    rows: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    for index in selected:
        axes[0].plot(gamma, curves[index], marker="o", markersize=2.5, linewidth=1.1, label=f"seed {seeds[index]}")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set(
        xlabel=r"$\Gamma_0$",
        ylabel=r"signed paper-distance $C_{\rm stag}$",
        title="(a) Physically selected seed trajectories",
    )
    axes[0].legend(frameon=False, fontsize=7, ncol=2)

    chosen = curves[selected]
    mean, sem = mean_sem(chosen)
    axes[1].plot(gamma, mean, "o-", color="black", markersize=3.5, label="selected-seed mean")
    axes[1].fill_between(gamma, mean - sem, mean + sem, color="black", alpha=0.16, label="SEM")
    axes[1].axhline(0.0, color="0.45", linewidth=0.8)
    selected_rows = [row for row in rows if row["selected"]]
    entry = np.asarray([row["entry_crossing"] for row in selected_rows], dtype=float)
    exit_ = np.asarray([row["exit_crossing"] for row in selected_rows], dtype=float)
    add_boundary_guides(axes[1], entry, exit_)
    axes[1].set(
        xlabel=r"$\Gamma_0$",
        ylabel=r"signed paper-distance $C_{\rm stag}$",
        title="(b) Mean after topology-blind selection",
    )
    axes[1].legend(frameon=False, fontsize=8)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"physical_seed_selection.{extension}", dpi=220)
    plt.close(figure)


def panel_specification() -> list[tuple[str, str, int | None]]:
    return [
        ("H0_velocity", r"$H_0$ landscape velocity", None),
        ("H0_death_range", r"$H_0$ death range", 3),
        ("H0_entropy", r"$H_0$ persistent entropy$/\log(15)$", 2),
        ("H0_max_lifetime", r"$H_0$ maximum lifetime", 0),
        ("H1_velocity", r"$H_1$ landscape velocity", None),
        ("H1_max_lifetime", r"$H_1$ maximum lifetime", 0),
        ("H1_total_persistence", r"$H_1$ total persistence", 1),
        ("H1_entropy", r"$H_1$ persistent entropy", 2),
    ]


def plot_individual_topology(
    output: Path,
    gamma: np.ndarray,
    gamma_mid: np.ndarray,
    seeds: np.ndarray,
    selected: np.ndarray,
    h0_scalar: np.ndarray,
    h1_scalar: np.ndarray,
    h0_individual_velocity: np.ndarray,
    h1_individual_velocity: np.ndarray,
    entry: np.ndarray,
    exit_: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(15.0, 7.4), sharex=False, constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, selected.size))
    for axis, (key, label, scalar_index) in zip(axes.flat, panel_specification(), strict=True):
        for color, seed_index in zip(colors, selected, strict=True):
            if key == "H0_velocity":
                x, values = gamma_mid, h0_individual_velocity[seed_index]
            elif key == "H1_velocity":
                x, values = gamma_mid, h1_individual_velocity[seed_index]
            elif key.startswith("H0"):
                x, values = gamma, h0_scalar[seed_index, :, scalar_index]
                if key == "H0_entropy":
                    values = values / np.log(15.0)
            else:
                x, values = gamma, h1_scalar[seed_index, :, scalar_index]
            axis.plot(x, values, color=color, linewidth=1.05, alpha=0.9, label=f"seed {seeds[seed_index]}")
        add_boundary_guides(axis, entry, exit_)
        axis.set(xlabel=r"$\Gamma_0$", ylabel=label)
        axis.grid(alpha=0.15)
    axes[0, 0].legend(frameon=False, fontsize=6.5, ncol=2)
    figure.suptitle("Individual topological responses of physically reentrant seeds", fontsize=13)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"selected_seed_topological_responses_individual.{extension}", dpi=220)
    plt.close(figure)


def plot_average_topology(
    output: Path,
    gamma: np.ndarray,
    gamma_mid: np.ndarray,
    selected: np.ndarray,
    h0_scalar: np.ndarray,
    h1_scalar: np.ndarray,
    h0_cohort_velocity: np.ndarray,
    h1_cohort_velocity: np.ndarray,
    entry: np.ndarray,
    exit_: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(15.0, 7.4), sharex=False, constrained_layout=True)
    for axis, (key, label, scalar_index) in zip(axes.flat, panel_specification(), strict=True):
        if key == "H0_velocity":
            x, mean, sem = gamma_mid, h0_cohort_velocity, None
        elif key == "H1_velocity":
            x, mean, sem = gamma_mid, h1_cohort_velocity, None
        elif key.startswith("H0"):
            values = h0_scalar[selected, :, scalar_index]
            if key == "H0_entropy":
                values = values / np.log(15.0)
            x = gamma
            mean, sem = mean_sem(values)
        else:
            x = gamma
            mean, sem = mean_sem(h1_scalar[selected, :, scalar_index])
        axis.plot(x, mean, "o-", color="black", markersize=3.0, linewidth=1.4)
        if sem is not None:
            axis.fill_between(x, mean - sem, mean + sem, color="black", alpha=0.16)
        add_boundary_guides(axis, entry, exit_)
        axis.set(xlabel=r"$\Gamma_0$", ylabel=label)
        axis.grid(alpha=0.15)
    figure.suptitle(
        "Selected-cohort topology (landscape: derivative of mean; scalars: mean ± SEM)",
        fontsize=13,
    )
    for extension in ("png", "pdf"):
        figure.savefig(output / f"selected_seed_topological_responses_average.{extension}", dpi=220)
    plt.close(figure)


def main() -> None:
    args = arguments()
    campaign = args.campaign.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else campaign / "analysis"
    )
    output.mkdir(parents=True, exist_ok=True)

    ensemble_path = campaign / "analysis" / "ensemble_summary.npz"
    scalar_path = campaign / "analysis" / "pbc_h0_h1_final" / "pbc_L16_final_h0_h1_arrays.npz"
    gamma_path = campaign / "analysis" / "pbc_gamma_velocity" / "arrays.npz"
    for path in (ensemble_path, scalar_path, gamma_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(ensemble_path, allow_pickle=False) as archive:
        h_values = np.asarray(archive["h_values"], dtype=float)
        h_index = int(np.argmin(np.abs(h_values - args.h)))
        if not np.isclose(h_values[h_index], args.h, atol=1.0e-10, rtol=0.0):
            raise ValueError(f"requested h={args.h} is absent; nearest is {h_values[h_index]}")
        gamma = np.asarray(archive["Gamma0_values"], dtype=float)
        seeds = np.asarray(archive["seeds"], dtype=int)
        physical = np.asarray(archive["paper_distance_Cstag_by_seed"][h_index], dtype=float)
        branch_disagreement = np.asarray(
            archive["branch_disagreement_by_seed"][h_index], dtype=bool
        )
        bond_saturation = np.asarray(
            archive["bond_saturation_by_seed"][h_index], dtype=bool
        )

    with np.load(scalar_path, allow_pickle=False) as archive:
        if not np.array_equal(seeds, archive["seeds"]) or not np.array_equal(gamma, archive["Gamma0"]):
            raise ValueError("ensemble and scalar topology archives are not aligned")
        h_scalar = np.asarray(archive["h"], dtype=float)
        scalar_h_index = int(np.argmin(np.abs(h_scalar - args.h)))
        h0_scalar = np.asarray(archive["H0_scalar"][scalar_h_index], dtype=float)
        h1_scalar = np.asarray(archive["H1_scalar"][scalar_h_index], dtype=float)

    with np.load(gamma_path, allow_pickle=False) as archive:
        if not np.array_equal(seeds, archive["seeds"]) or not np.array_equal(gamma, archive["Gamma0"]):
            raise ValueError("ensemble and Gamma-topology archives are not aligned")
        h_gamma = np.asarray(archive["h"], dtype=float)
        gamma_h_index = int(np.argmin(np.abs(h_gamma - args.h)))
        gamma_mid = np.asarray(archive["Gamma0_mid"], dtype=float)
        h0_gram = np.asarray(archive["H0_grams"][gamma_h_index], dtype=float)
        h1_gram = np.asarray(archive["H1_grams"][gamma_h_index], dtype=float)

    selection_rows, selected = select_physical_seeds(
        gamma,
        seeds,
        physical,
        endpoint_points=args.endpoint_points,
        negative_margin=args.negative_margin,
        positive_amplitude=args.positive_amplitude,
        minimum_lobe_width=args.minimum_lobe_width,
        minimum_positive_run=args.minimum_positive_run,
    )
    if selected.size == 0:
        raise RuntimeError("the physical rule selected no seeds")
    selected_rows = [row for row in selection_rows if row["selected"]]
    for seed_index, row in enumerate(selection_rows):
        row["branch_disagreement_count"] = int(np.count_nonzero(branch_disagreement[seed_index]))
        row["bond_saturation_count"] = int(np.count_nonzero(bond_saturation[seed_index]))
    entry = np.asarray([row["entry_crossing"] for row in selected_rows], dtype=float)
    exit_ = np.asarray([row["exit_crossing"] for row in selected_rows], dtype=float)

    h0_individual_velocity = np.sqrt(np.maximum(np.diagonal(h0_gram, axis1=1, axis2=2), 0.0)).T
    h1_individual_velocity = np.sqrt(np.maximum(np.diagonal(h1_gram, axis1=1, axis2=2), 0.0)).T
    h0_cohort_velocity = speed_from_gram(h0_gram, selected)
    h1_cohort_velocity = speed_from_gram(h1_gram, selected)

    peak_rows: list[dict[str, object]] = []
    h0_entry_errors: list[float] = []
    h1_exit_errors: list[float] = []
    for selected_position, seed_index in enumerate(selected):
        h0_peak_index = int(np.argmax(h0_individual_velocity[seed_index]))
        h1_peak_index = int(np.argmax(h1_individual_velocity[seed_index]))
        h0_peak = float(gamma_mid[h0_peak_index])
        h1_peak = float(gamma_mid[h1_peak_index])
        h0_error = h0_peak - float(entry[selected_position])
        h1_error = h1_peak - float(exit_[selected_position])
        h0_entry_errors.append(h0_error)
        h1_exit_errors.append(h1_error)
        peak_rows.append(
            {
                "seed": int(seeds[seed_index]),
                "physical_entry": float(entry[selected_position]),
                "H0_global_velocity_peak": h0_peak,
                "H0_peak_minus_entry": h0_error,
                "physical_exit": float(exit_[selected_position]),
                "H1_global_velocity_peak": h1_peak,
                "H1_peak_minus_exit": h1_error,
            }
        )

    write_csv(output / "physical_seed_selection.csv", selection_rows)
    write_csv(output / "topological_peak_diagnostics.csv", peak_rows)
    curve_rows: list[dict[str, object]] = []
    for seed_index in selected:
        for gamma_index, gamma_value in enumerate(gamma):
            row: dict[str, object] = {
                "seed": int(seeds[seed_index]),
                "Gamma0": float(gamma_value),
                "paper_distance_Cstag": float(physical[seed_index, gamma_index]),
            }
            for dimension, scalar_values in ((0, h0_scalar), (1, h1_scalar)):
                for scalar_index, name in enumerate(SCALAR_NAMES):
                    row[f"H{dimension}_{name}"] = float(scalar_values[seed_index, gamma_index, scalar_index])
            curve_rows.append(row)
    write_csv(output / "selected_seed_scalar_curves.csv", curve_rows)

    velocity_rows: list[dict[str, object]] = []
    for seed_index in selected:
        for gamma_index, gamma_value in enumerate(gamma_mid):
            velocity_rows.append(
                {
                    "seed": int(seeds[seed_index]),
                    "Gamma0_mid": float(gamma_value),
                    "H0_individual_landscape_velocity": float(h0_individual_velocity[seed_index, gamma_index]),
                    "H1_individual_landscape_velocity": float(h1_individual_velocity[seed_index, gamma_index]),
                }
            )
    write_csv(output / "selected_seed_landscape_velocity.csv", velocity_rows)

    average_rows: list[dict[str, object]] = []
    for gamma_index, gamma_value in enumerate(gamma):
        row = {"Gamma0": float(gamma_value)}
        for dimension, scalar_values in ((0, h0_scalar), (1, h1_scalar)):
            for scalar_index, name in enumerate(SCALAR_NAMES):
                mean, sem = mean_sem(scalar_values[selected, :, scalar_index])
                row[f"H{dimension}_{name}_mean"] = float(mean[gamma_index])
                row[f"H{dimension}_{name}_sem"] = finite_or_none(float(sem[gamma_index]))
        average_rows.append(row)
    write_csv(output / "selected_cohort_scalar_average.csv", average_rows)
    write_csv(
        output / "selected_cohort_landscape_velocity.csv",
        [
            {
                "Gamma0_mid": float(value),
                "H0_mean_landscape_velocity": float(h0_cohort_velocity[index]),
                "H1_mean_landscape_velocity": float(h1_cohort_velocity[index]),
            }
            for index, value in enumerate(gamma_mid)
        ],
    )

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "selected_seed_topology_arrays.npz",
        h=np.asarray(args.h),
        Gamma0=gamma,
        Gamma0_mid=gamma_mid,
        seeds=seeds,
        selected_seed_indices=selected,
        selected_seeds=seeds[selected],
        physical_paper_distance_Cstag=physical,
        entry_crossings=entry,
        exit_crossings=exit_,
        H0_scalar=h0_scalar,
        H1_scalar=h1_scalar,
        H0_individual_landscape_velocity=h0_individual_velocity,
        H1_individual_landscape_velocity=h1_individual_velocity,
        H0_selected_mean_landscape_velocity=h0_cohort_velocity,
        H1_selected_mean_landscape_velocity=h1_cohort_velocity,
    )

    plot_physical_selection(output, gamma, seeds, physical, selected, selection_rows)
    plot_individual_topology(
        output,
        gamma,
        gamma_mid,
        seeds,
        selected,
        h0_scalar,
        h1_scalar,
        h0_individual_velocity,
        h1_individual_velocity,
        entry,
        exit_,
    )
    plot_average_topology(
        output,
        gamma,
        gamma_mid,
        selected,
        h0_scalar,
        h1_scalar,
        h0_cohort_velocity,
        h1_cohort_velocity,
        entry,
        exit_,
    )

    summary = {
        "status": "complete_exploratory",
        "campaign": campaign.name,
        "conditions": {
            "L": 16,
            "h": float(args.h),
            "bc": "periodic",
            "ensemble": "fixed_mean",
            "matrix": "C=Cstag",
            "filtration": "D=1-C formed by the upstream topology analysis",
        },
        "selection_is_topology_blind": True,
        "physical_selection": {
            "observable": "signed paper-distance Cstag",
            "rule": "negative-positive-negative curve with exactly two linearly interpolated zero crossings",
            "endpoint_points": int(args.endpoint_points),
            "negative_margin": float(args.negative_margin),
            "positive_amplitude": float(args.positive_amplitude),
            "minimum_lobe_width": float(args.minimum_lobe_width),
            "minimum_positive_run": int(args.minimum_positive_run),
            "selected_seed_count": int(selected.size),
            "selected_seeds": seeds[selected].tolist(),
            "entry_crossings": entry.tolist(),
            "exit_crossings": exit_.tolist(),
            "entry_median": float(np.median(entry)),
            "entry_range": [float(np.min(entry)), float(np.max(entry))],
            "exit_median": float(np.median(exit_)),
            "exit_range": [float(np.min(exit_)), float(np.max(exit_))],
        },
        "topological_measures": {
            "H0_and_H1_landscape_velocity_individual": "sqrt of the diagonal of each Gamma-interval Gram matrix",
            "H0_and_H1_selected_cohort_landscape_velocity": "L2 norm of the Gamma derivative of the selected-seed mean landscape",
            "scalar_order": list(SCALAR_NAMES),
            "random_h0_measures": ["H0 landscape velocity", "H0 death range", "H0 persistent entropy"],
            "additional_diagnostics": ["H0 maximum lifetime", "H1 landscape velocity", "H1 maximum lifetime", "H1 total persistence", "H1 persistent entropy"],
        },
        "autonomous_global_velocity_peak_diagnostics": {
            "definition": "global maximum over the complete Gamma0 interval; no nearest-physical-peak selection",
            "H0_peak_vs_physical_entry": {
                "mean_signed_error": float(np.mean(h0_entry_errors)),
                "mean_absolute_error": float(np.mean(np.abs(h0_entry_errors))),
                "per_seed": peak_rows,
            },
            "H1_peak_vs_physical_exit": {
                "mean_signed_error": float(np.mean(h1_exit_errors)),
                "mean_absolute_error": float(np.mean(np.abs(h1_exit_errors))),
                "per_seed": peak_rows,
            },
            "selected_cohort_H0_global_peak": float(gamma_mid[int(np.argmax(h0_cohort_velocity))]),
            "selected_cohort_H1_global_peak": float(gamma_mid[int(np.argmax(h1_cohort_velocity))]),
        },
        "qc_selected_cohort": {
            "branch_disagreement_points": int(np.count_nonzero(branch_disagreement[selected])),
            "bond_saturation_points": int(np.count_nonzero(bond_saturation[selected])),
            "total_points": int(selected.size * gamma.size),
            "selection_did_not_filter_on_qc": True,
        },
        "source_sha256": {
            str(ensemble_path.relative_to(ROOT)): sha256(ensemble_path),
            str(scalar_path.relative_to(ROOT)): sha256(scalar_path),
            str(gamma_path.relative_to(ROOT)): sha256(gamma_path),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "selected_seed_topology_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "SELECTED_SEED_TOPOLOGY_README.md").write_text(
        "# Selected-seed PBC topology at h=2.5\n\n"
        "Exploratory, topology-blind selection of PBC L=16 TN seeds with a clear "
        "signed physical reentrant lobe, followed by individual and selected-cohort "
        "H0/H1 persistent-homology responses.\n\n"
        "The landscape aggregate is the derivative norm of the selected-seed mean "
        "landscape, matching the random h=0 convention. Scalar summaries are averaged "
        "after being computed per seed. See `selected_seed_topology_summary.json` and "
        "`physical_seed_selection.csv` for the frozen rule and complete audit.\n",
        encoding="utf-8",
    )
    print(f"selected seeds: {seeds[selected].tolist()}")
    print(f"entry median/range: {np.median(entry):.6f} [{np.min(entry):.6f}, {np.max(entry):.6f}]")
    print(f"exit median/range: {np.median(exit_):.6f} [{np.min(exit_):.6f}, {np.max(exit_):.6f}]")
    print(f"analysis written to: {output}")


if __name__ == "__main__":
    main()
