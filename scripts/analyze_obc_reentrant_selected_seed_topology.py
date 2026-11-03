"""Topology of physically selected reentrant OBC L=16 disorder samples.

The physical cohort is frozen without topology: at a fixed longitudinal field,
the signed paper-distance Cstag curve must follow a negative-positive-negative
trajectory with exactly two robust zero crossings.  Persistent homology is
then computed directly from C=Cstag (with D=1-C inside this script) for only
the selected samples.  This keeps selection and evaluation strictly separate.

The three input ensemble archives together contain the complete OBC L=16
cohort, seeds 0...99.  Outputs are exploratory and stay under results/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_pbc_reentrant_selected_seed_topology import (  # noqa: E402
    mean_sem,
    plot_average_topology,
    plot_individual_topology,
    plot_physical_selection,
    select_physical_seeds,
    write_csv,
)


BASE = ROOT / "results" / "random_ising_gamma_ensemble"
CAMPAIGNS = (
    "L16-reentrant-map-targeted-expanded-s0-49-v1",
    "L16-reentrant-holdout-c32-s50-74-v1",
    "L16-reentrant-holdout-c32-s75-99-turing-merged-v1",
)
DEFAULT_OUTPUT = (
    BASE
    / "L16-reentrant-obc-selected-seed-topology-s0-99-v1"
    / "analysis"
)
GRID = np.linspace(0.0, 2.0, 401)
LAYERS = 15
SCALAR_NAMES = (
    "maximum_lifetime",
    "total_persistence",
    "persistent_entropy",
    "death_range",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h", type=float, default=2.5)
    parser.add_argument("--endpoint-points", type=int, default=2)
    parser.add_argument("--negative-margin", type=float, default=0.02)
    parser.add_argument("--positive-amplitude", type=float, default=0.05)
    parser.add_argument("--minimum-lobe-width", type=float, default=0.10)
    parser.add_argument("--minimum-positive-run", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_complete_cohort() -> tuple[dict[str, np.ndarray], list[Path]]:
    parts: list[dict[str, np.ndarray]] = []
    paths: list[Path] = []
    for campaign in CAMPAIGNS:
        path = BASE / campaign / "analysis" / "ensemble_summary.npz"
        paths.append(path)
        with np.load(path, allow_pickle=False) as archive:
            parts.append(
                {
                    "h": np.asarray(archive["h_values"], dtype=float),
                    "gamma": np.asarray(archive["Gamma0_values"], dtype=float),
                    "seeds": np.asarray(archive["seeds"], dtype=int),
                    "physical": np.asarray(
                        archive["paper_distance_Cstag_by_seed"], dtype=float
                    ),
                    "Cstag": np.asarray(archive["Cstag_by_seed"], dtype=float),
                    "branch": np.asarray(
                        archive["branch_disagreement_by_seed"], dtype=bool
                    ),
                    "saturated": np.asarray(
                        archive["bond_saturation_by_seed"], dtype=bool
                    ),
                }
            )
    for part in parts[1:]:
        if not np.array_equal(parts[0]["h"], part["h"]):
            raise ValueError("input h grids differ")
        if not np.array_equal(parts[0]["gamma"], part["gamma"]):
            raise ValueError("input Gamma0 grids differ")
    seeds = np.concatenate([part["seeds"] for part in parts])
    order = np.argsort(seeds)
    if not np.array_equal(seeds[order], np.arange(100)):
        raise ValueError("the combined cohort must contain seeds 0...99 exactly once")
    output = {
        "h": parts[0]["h"],
        "gamma": parts[0]["gamma"],
        "seeds": seeds[order],
    }
    for key in ("physical", "Cstag", "branch", "saturated"):
        output[key] = np.concatenate([part[key] for part in parts], axis=1)[:, order]
    return output, paths


def persistence(cstag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import gudhi

    correlation = 0.5 * (np.asarray(cstag, dtype=float) + np.asarray(cstag, dtype=float).T)
    distance = np.clip(1.0 - correlation, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    tree = gudhi.RipsComplex(
        distance_matrix=distance, max_edge_length=2.0
    ).create_simplex_tree(max_dimension=2)
    tree.persistence(homology_coeff_field=2, min_persistence=-1.0)
    diagrams: list[np.ndarray] = []
    for dimension in (0, 1):
        diagram = np.asarray(
            tree.persistence_intervals_in_dimension(dimension), dtype=float
        )
        if diagram.size == 0:
            diagram = np.empty((0, 2), dtype=float)
        lifetime = diagram[:, 1] - diagram[:, 0]
        finite = diagram[np.isfinite(lifetime) & (lifetime > 1.0e-14)]
        diagrams.append(finite)
    return diagrams[0], diagrams[1]


def scalar_summary(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return np.zeros(4, dtype=float)
    lifetime = diagram[:, 1] - diagram[:, 0]
    total = float(np.sum(lifetime))
    probability = lifetime / total
    return np.asarray(
        [
            np.max(lifetime),
            total,
            -np.sum(probability * np.log(probability)),
            np.ptp(diagram[:, 1]),
        ],
        dtype=float,
    )


def landscape(diagram: np.ndarray) -> np.ndarray:
    result = np.zeros((LAYERS, GRID.size), dtype=np.float32)
    if diagram.size == 0:
        return result
    tents = np.maximum(
        np.minimum(
            GRID[np.newaxis, :] - diagram[:, 0, np.newaxis],
            diagram[:, 1, np.newaxis] - GRID[np.newaxis, :],
        ),
        0.0,
    )
    ranked = np.sort(tents, axis=0)[::-1]
    count = min(LAYERS, ranked.shape[0])
    result[:count] = ranked[:count]
    return result


def landscape_velocity(values: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    derivative = np.diff(values, axis=1) / np.diff(gamma)[None, :, None, None]
    return np.sqrt(
        np.trapezoid(np.sum(derivative.astype(float) ** 2, axis=2), GRID, axis=2)
        / LAYERS
    )


def cohort_velocity(values: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    mean_landscape = np.mean(values.astype(float), axis=0)
    derivative = np.diff(mean_landscape, axis=0) / np.diff(gamma)[:, None, None]
    return np.sqrt(
        np.trapezoid(np.sum(derivative**2, axis=1), GRID, axis=1) / LAYERS
    )


def compute_topology(cstag: np.ndarray) -> dict[str, np.ndarray]:
    seed_count, gamma_count = cstag.shape[:2]
    scalar = {
        0: np.empty((seed_count, gamma_count, 4), dtype=float),
        1: np.empty((seed_count, gamma_count, 4), dtype=float),
    }
    landscapes = {
        0: np.empty((seed_count, gamma_count, LAYERS, GRID.size), dtype=np.float32),
        1: np.empty((seed_count, gamma_count, LAYERS, GRID.size), dtype=np.float32),
    }
    for seed_index in range(seed_count):
        for gamma_index in range(gamma_count):
            diagrams = persistence(cstag[seed_index, gamma_index])
            for dimension in (0, 1):
                scalar[dimension][seed_index, gamma_index] = scalar_summary(
                    diagrams[dimension]
                )
                landscapes[dimension][seed_index, gamma_index] = landscape(
                    diagrams[dimension]
                )
        print(f"topology selected seed {seed_index + 1}/{seed_count}", flush=True)
    return {
        "H0_scalar": scalar[0],
        "H1_scalar": scalar[1],
        "H0_landscape": landscapes[0],
        "H1_landscape": landscapes[1],
    }


def plot_selection_stability(
    output: Path,
    h: np.ndarray,
    seeds: np.ndarray,
    selected_by_h: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    ever = np.flatnonzero(np.any(selected_by_h, axis=0))
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    image = axes[0].imshow(
        selected_by_h[:, ever].T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[h[0], h[-1], -0.5, ever.size - 0.5],
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    del image
    axes[0].set_yticks(np.arange(ever.size), labels=seeds[ever])
    axes[0].axvline(2.5, color="#D55E00", linestyle="--", linewidth=1.2)
    axes[0].set(
        xlabel=r"$h$",
        ylabel="seed selected at least once",
        title="Topology-blind physical selection",
    )
    axes[1].plot(h, np.sum(selected_by_h, axis=1), "o-", color="black")
    axes[1].axvline(2.5, color="#D55E00", linestyle="--", linewidth=1.2)
    axes[1].set(
        xlabel=r"$h$",
        ylabel="selected-seed count",
        title="Selection count versus field",
    )
    axes[1].grid(alpha=0.2)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"physical_selection_h_stability.{extension}", dpi=220)
    plt.close(figure)


def main() -> None:
    args = arguments()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data, source_paths = load_complete_cohort()
    h = data["h"]
    gamma = data["gamma"]
    seeds = data["seeds"]
    h_index = int(np.argmin(np.abs(h - args.h)))
    if not np.isclose(h[h_index], args.h, atol=1.0e-12, rtol=0.0):
        raise ValueError(f"requested h={args.h} is absent")

    selection_kwargs = {
        "endpoint_points": args.endpoint_points,
        "negative_margin": args.negative_margin,
        "positive_amplitude": args.positive_amplitude,
        "minimum_lobe_width": args.minimum_lobe_width,
        "minimum_positive_run": args.minimum_positive_run,
    }
    rows, selected = select_physical_seeds(
        gamma, seeds, data["physical"][h_index], **selection_kwargs
    )
    if selected.size == 0:
        raise RuntimeError("the physical rule selected no OBC seeds")
    for seed_index, row in enumerate(rows):
        row["cohort"] = "reference_0_49" if seeds[seed_index] < 50 else "holdout_50_99"
        row["branch_disagreement_count"] = int(
            np.count_nonzero(data["branch"][h_index, seed_index])
        )
        row["bond_saturation_count"] = int(
            np.count_nonzero(data["saturated"][h_index, seed_index])
        )
    selected_rows = [row for row in rows if row["selected"]]
    entry = np.asarray([row["entry_crossing"] for row in selected_rows], dtype=float)
    exit_ = np.asarray([row["exit_crossing"] for row in selected_rows], dtype=float)

    selected_by_h = np.zeros((h.size, seeds.size), dtype=bool)
    stability_rows: list[dict[str, object]] = []
    for current_h_index, h_value in enumerate(h):
        current_rows, current_selected = select_physical_seeds(
            gamma, seeds, data["physical"][current_h_index], **selection_kwargs
        )
        selected_by_h[current_h_index, current_selected] = True
        chosen = {int(seeds[index]) for index in current_selected}
        for row in current_rows:
            if row["selected"]:
                stability_rows.append(
                    {
                        "h": float(h_value),
                        "seed": int(row["seed"]),
                        "entry_crossing": float(row["entry_crossing"]),
                        "exit_crossing": float(row["exit_crossing"]),
                    }
                )
        print(f"h={h_value:g}: selected {len(chosen)}", flush=True)

    topology = compute_topology(data["Cstag"][h_index, selected])
    gamma_mid = 0.5 * (gamma[:-1] + gamma[1:])
    h0_individual_velocity = landscape_velocity(topology["H0_landscape"], gamma)
    h1_individual_velocity = landscape_velocity(topology["H1_landscape"], gamma)
    h0_cohort_velocity = cohort_velocity(topology["H0_landscape"], gamma)
    h1_cohort_velocity = cohort_velocity(topology["H1_landscape"], gamma)

    # Gamma0=0 is exactly classical and can be degenerate in OBC.  Preserve
    # its interval in all arrays, but exclude 0 -> 0.05 from primary peak tests.
    primary_velocity = np.flatnonzero(gamma[:-1] >= 0.05 - 1.0e-12)
    peak_rows: list[dict[str, object]] = []
    h0_errors: list[float] = []
    h1_errors: list[float] = []
    for local_index, global_index in enumerate(selected):
        h0_peak_index = primary_velocity[
            int(np.argmax(h0_individual_velocity[local_index, primary_velocity]))
        ]
        h1_peak_index = primary_velocity[
            int(np.argmax(h1_individual_velocity[local_index, primary_velocity]))
        ]
        h0_peak = float(gamma_mid[h0_peak_index])
        h1_peak = float(gamma_mid[h1_peak_index])
        h0_error = h0_peak - float(entry[local_index])
        h1_error = h1_peak - float(exit_[local_index])
        h0_errors.append(h0_error)
        h1_errors.append(h1_error)
        peak_rows.append(
            {
                "seed": int(seeds[global_index]),
                "cohort": "reference_0_49" if seeds[global_index] < 50 else "holdout_50_99",
                "physical_entry": float(entry[local_index]),
                "H0_primary_velocity_peak": h0_peak,
                "H0_peak_minus_entry": h0_error,
                "physical_exit": float(exit_[local_index]),
                "H1_primary_velocity_peak": h1_peak,
                "H1_peak_minus_exit": h1_error,
                "excluded_classical_H0_velocity": float(h0_individual_velocity[local_index, 0]),
                "excluded_classical_H1_velocity": float(h1_individual_velocity[local_index, 0]),
            }
        )

    write_csv(output / "physical_seed_selection.csv", rows)
    write_csv(output / "physical_selection_h_stability.csv", stability_rows)
    write_csv(output / "topological_peak_diagnostics.csv", peak_rows)

    curve_rows: list[dict[str, object]] = []
    for local_index, global_index in enumerate(selected):
        for gamma_index, gamma_value in enumerate(gamma):
            row: dict[str, object] = {
                "seed": int(seeds[global_index]),
                "Gamma0": float(gamma_value),
                "paper_distance_Cstag": float(data["physical"][h_index, global_index, gamma_index]),
            }
            for dimension in (0, 1):
                values = topology[f"H{dimension}_scalar"]
                for scalar_index, name in enumerate(SCALAR_NAMES):
                    row[f"H{dimension}_{name}"] = float(
                        values[local_index, gamma_index, scalar_index]
                    )
            curve_rows.append(row)
    write_csv(output / "selected_seed_scalar_curves.csv", curve_rows)

    velocity_rows: list[dict[str, object]] = []
    for local_index, global_index in enumerate(selected):
        for gamma_index, gamma_value in enumerate(gamma_mid):
            velocity_rows.append(
                {
                    "seed": int(seeds[global_index]),
                    "Gamma0_mid": float(gamma_value),
                    "classical_interval_excluded_from_primary_peak": int(gamma_index == 0),
                    "H0_individual_landscape_velocity": float(h0_individual_velocity[local_index, gamma_index]),
                    "H1_individual_landscape_velocity": float(h1_individual_velocity[local_index, gamma_index]),
                }
            )
    write_csv(output / "selected_seed_landscape_velocity.csv", velocity_rows)

    average_rows: list[dict[str, object]] = []
    all_local = np.arange(selected.size)
    for gamma_index, gamma_value in enumerate(gamma):
        row = {"Gamma0": float(gamma_value)}
        for dimension in (0, 1):
            values = topology[f"H{dimension}_scalar"]
            for scalar_index, name in enumerate(SCALAR_NAMES):
                mean, sem = mean_sem(values[:, :, scalar_index])
                row[f"H{dimension}_{name}_mean"] = float(mean[gamma_index])
                row[f"H{dimension}_{name}_sem"] = (
                    float(sem[gamma_index]) if np.isfinite(sem[gamma_index]) else ""
                )
        average_rows.append(row)
    write_csv(output / "selected_cohort_scalar_average.csv", average_rows)
    write_csv(
        output / "selected_cohort_landscape_velocity.csv",
        [
            {
                "Gamma0_mid": float(value),
                "classical_interval_excluded_from_primary_peak": int(index == 0),
                "H0_mean_landscape_velocity": float(h0_cohort_velocity[index]),
                "H1_mean_landscape_velocity": float(h1_cohort_velocity[index]),
            }
            for index, value in enumerate(gamma_mid)
        ],
    )

    np.savez_compressed(
        output / "selected_seed_topology_arrays.npz",
        h=np.asarray(args.h),
        Gamma0=gamma,
        Gamma0_mid=gamma_mid,
        landscape_grid=GRID,
        landscape_layers=np.asarray(LAYERS),
        seeds=seeds,
        selected_seed_indices=selected,
        selected_seeds=seeds[selected],
        physical_paper_distance_Cstag=data["physical"][h_index],
        selection_by_h=selected_by_h,
        entry_crossings=entry,
        exit_crossings=exit_,
        H0_scalar=topology["H0_scalar"],
        H1_scalar=topology["H1_scalar"],
        H0_individual_landscape_velocity=h0_individual_velocity,
        H1_individual_landscape_velocity=h1_individual_velocity,
        H0_selected_mean_landscape_velocity=h0_cohort_velocity,
        H1_selected_mean_landscape_velocity=h1_cohort_velocity,
    )

    plot_physical_selection(
        output, gamma, seeds, data["physical"][h_index], selected, rows
    )
    plot_individual_topology(
        output,
        gamma,
        gamma_mid,
        seeds[selected],
        all_local,
        topology["H0_scalar"],
        topology["H1_scalar"],
        h0_individual_velocity,
        h1_individual_velocity,
        entry,
        exit_,
    )
    plot_average_topology(
        output,
        gamma,
        gamma_mid,
        all_local,
        topology["H0_scalar"],
        topology["H1_scalar"],
        h0_cohort_velocity,
        h1_cohort_velocity,
        entry,
        exit_,
    )
    plot_selection_stability(output, h, seeds, selected_by_h)

    cohort_h0_peak_index = primary_velocity[
        int(np.argmax(h0_cohort_velocity[primary_velocity]))
    ]
    cohort_h1_peak_index = primary_velocity[
        int(np.argmax(h1_cohort_velocity[primary_velocity]))
    ]
    summary = {
        "status": "complete_exploratory",
        "conditions": {
            "L": 16,
            "h": float(args.h),
            "bc": "open",
            "chi": 32,
            "ensemble": "fixed_mean",
            "matrix": "C=Cstag",
            "filtration": "D=1-C formed inside this analysis",
        },
        "input_cohort": {
            "seed_count": int(seeds.size),
            "seeds": [int(seeds[0]), int(seeds[-1])],
            "h_count": int(h.size),
            "h_range": [float(h[0]), float(h[-1])],
            "Gamma0_count": int(gamma.size),
            "Gamma0_range": [float(gamma[0]), float(gamma[-1])],
            "campaigns": list(CAMPAIGNS),
        },
        "selection_is_topology_blind": True,
        "physical_selection": {
            "observable": "signed paper-distance Cstag",
            "rule": "negative-positive-negative with exactly two linearly interpolated zero crossings",
            **selection_kwargs,
            "selected_seed_count": int(selected.size),
            "selected_seeds": seeds[selected].tolist(),
            "reference_selected": seeds[selected][seeds[selected] < 50].tolist(),
            "holdout_selected": seeds[selected][seeds[selected] >= 50].tolist(),
            "entry_crossings": entry.tolist(),
            "entry_median": float(np.median(entry)),
            "entry_range": [float(np.min(entry)), float(np.max(entry))],
            "exit_crossings": exit_.tolist(),
            "exit_median": float(np.median(exit_)),
            "exit_range": [float(np.min(exit_)), float(np.max(exit_))],
        },
        "h_stability": {
            "selection_counts": {
                f"{value:.6g}": int(np.count_nonzero(selected_by_h[index]))
                for index, value in enumerate(h)
            },
            "primary_selection_unchanged_at": [
                float(h[index])
                for index in range(h.size)
                if np.array_equal(selected_by_h[index], selected_by_h[h_index])
            ],
        },
        "topology": {
            "scalar_order": list(SCALAR_NAMES),
            "landscape_grid": [float(GRID[0]), float(GRID[-1]), int(GRID.size)],
            "landscape_layers": LAYERS,
            "velocity_normalization": "L2 norm divided by sqrt(15)",
            "cohort_velocity": "Gamma0 derivative of the selected-seed mean landscape",
            "scalar_aggregate": "mean and SEM after computing each seed separately",
        },
        "classical_endpoint_policy": {
            "Gamma0_zero_retained_in_curves": True,
            "interval_0_to_0p05_excluded_from_primary_velocity_peaks": True,
            "reason": "exact classical degeneracy can collapse OBC H0 summaries",
        },
        "primary_peak_diagnostics": {
            "H0_peak_vs_physical_entry": {
                "mean_signed_error": float(np.mean(h0_errors)),
                "mean_absolute_error": float(np.mean(np.abs(h0_errors))),
                "median_absolute_error": float(np.median(np.abs(h0_errors))),
                "within_one_Gamma0_step_count": int(
                    np.count_nonzero(np.abs(h0_errors) <= 0.05 + 1.0e-12)
                ),
                "selected_seed_count": int(selected.size),
            },
            "H1_peak_vs_physical_exit": {
                "mean_signed_error": float(np.mean(h1_errors)),
                "mean_absolute_error": float(np.mean(np.abs(h1_errors))),
                "median_absolute_error": float(np.median(np.abs(h1_errors))),
                "within_one_Gamma0_step_count": int(
                    np.count_nonzero(np.abs(h1_errors) <= 0.05 + 1.0e-12)
                ),
                "selected_seed_count": int(selected.size),
            },
            "selected_cohort_H0_peak": float(gamma_mid[cohort_h0_peak_index]),
            "selected_cohort_H1_peak": float(gamma_mid[cohort_h1_peak_index]),
            "selected_cohort_H0_peak_minus_entry_median": float(
                gamma_mid[cohort_h0_peak_index] - np.median(entry)
            ),
            "selected_cohort_H1_peak_minus_exit_median": float(
                gamma_mid[cohort_h1_peak_index] - np.median(exit_)
            ),
            "per_seed": peak_rows,
        },
        "qc_selected_cohort": {
            "branch_disagreement_points": int(
                np.count_nonzero(data["branch"][h_index, selected])
            ),
            "bond_saturation_points": int(
                np.count_nonzero(data["saturated"][h_index, selected])
            ),
            "total_points": int(selected.size * gamma.size),
            "selection_did_not_filter_on_qc": True,
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
    }
    (output / "selected_seed_topology_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# OBC L=16 physically selected seed topology\n\n"
        "Exploratory, topology-blind physical selection at h=2.5 over the complete "
        "100-seed OBC cohort, followed by individual and selected-cohort H0/H1 "
        "persistent-homology responses from C=Cstag. The first velocity interval "
        "touching Gamma0=0 is retained but excluded from primary peak diagnostics "
        "because of exact classical degeneracy.\n",
        encoding="utf-8",
    )
    print(f"selected seeds: {seeds[selected].tolist()}")
    print(
        f"entry median {np.median(entry):.6f}; exit median {np.median(exit_):.6f}"
    )
    print(f"analysis written to {output}")


if __name__ == "__main__":
    main()
