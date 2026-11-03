"""IRFP-motivated pseudocritical shift and width analysis for random h=0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DATA = ROOT / "paper_figures" / "fig_random_h0" / "data_raw"
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "random_ising_gamma_ensemble"
    / "random-h0-irfp-pseudocritical-scaling-v1"
    / "analysis"
)
EXACT_CRITICAL_GAMMA0 = 1.0
DEFAULT_BOOTSTRAPS = 5000
DEFAULT_RNG_SEED = 20260904
FIT_MINIMA = (8, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot summarize an empty finite array")
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "q025": float(np.quantile(finite, 0.025)),
        "q975": float(np.quantile(finite, 0.975)),
    }


def local_quadratic_peak(
    gamma: np.ndarray,
    values: np.ndarray,
) -> dict[str, float | int | bool | list[float]]:
    gamma = np.asarray(gamma, dtype=float)
    values = np.asarray(values, dtype=float)
    index = int(np.argmax(values))
    discrete_gamma = float(gamma[index])
    result: dict[str, float | int | bool | list[float]] = {
        "Gamma0": discrete_gamma,
        "value": float(values[index]),
        "discrete_index": index,
        "discrete_Gamma0": discrete_gamma,
        "used_quadratic": False,
        "edge_maximum": index in (0, gamma.size - 1),
        "fit_window_Gamma0": [discrete_gamma],
    }
    if bool(result["edge_maximum"]):
        return result

    window = slice(index - 1, index + 2)
    center = discrete_gamma
    coefficients = np.polyfit(
        gamma[window] - center,
        values[window],
        deg=2,
    )
    if coefficients[0] >= 0.0 or np.isclose(coefficients[0], 0.0):
        return result
    offset = float(-coefficients[1] / (2.0 * coefficients[0]))
    refined_gamma = center + offset
    if not gamma[index - 1] <= refined_gamma <= gamma[index + 1]:
        return result

    result.update(
        {
            "Gamma0": float(refined_gamma),
            "value": float(np.polyval(coefficients, offset)),
            "used_quadratic": True,
            "fit_window_Gamma0": [
                float(item) for item in gamma[window]
            ],
        }
    )
    return result


def peaks_for_curves(
    gamma: np.ndarray,
    curves: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2 or curves.shape[1] != gamma.size:
        raise ValueError("curves must have shape (n_curves, n_gamma)")
    details = [local_quadratic_peak(gamma, curve) for curve in curves]
    return (
        np.asarray([item["Gamma0"] for item in details], dtype=float),
        np.asarray(
            [item["discrete_Gamma0"] for item in details],
            dtype=float,
        ),
        np.asarray([item["edge_maximum"] for item in details], dtype=bool),
        np.asarray([item["used_quadratic"] for item in details], dtype=bool),
    )


def power_law_fit(
    sizes: np.ndarray,
    values: np.ndarray,
    *,
    minimum_size: int,
) -> dict[str, Any]:
    sizes = np.asarray(sizes, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = (
        (sizes >= minimum_size)
        & np.isfinite(values)
        & (values > 0.0)
    )
    if np.count_nonzero(mask) < 3:
        raise ValueError("at least three positive points are required")
    x = np.log(sizes[mask])
    y = np.log(values[mask])
    slope, intercept = np.polyfit(x, y, deg=1)
    prediction = slope * x + intercept
    residual = y - prediction
    total = y - np.mean(y)
    denominator = float(np.sum(total**2))
    return {
        "minimum_size": int(minimum_size),
        "sizes": [int(item) for item in sizes[mask]],
        "A": float(np.exp(intercept)),
        "p": float(-slope),
        "log_rmse": float(np.sqrt(np.mean(residual**2))),
        "log_r2": (
            float(1.0 - np.sum(residual**2) / denominator)
            if denominator > 0.0
            else float("nan")
        ),
    }


def bootstrap_power_exponents(
    sizes: np.ndarray,
    values: np.ndarray,
    *,
    minimum_size: int,
) -> np.ndarray:
    sizes = np.asarray(sizes, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = sizes >= minimum_size
    x = np.log(sizes[mask])
    output = np.full(values.shape[0], np.nan, dtype=float)
    for index, row in enumerate(values[:, mask]):
        finite = np.isfinite(row) & (row > 0.0)
        if np.count_nonzero(finite) >= 3:
            output[index] = -float(
                np.polyfit(x[finite], np.log(row[finite]), deg=1)[0]
            )
    return output


def landscape_gram_matrices(
    run: Path,
    gamma: np.ndarray,
    seeds: np.ndarray,
    L: int,
) -> tuple[np.ndarray, float]:
    with (run / "summaries.csv").open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["homology_dimension"]) == 0
            and row["status"] == "complete"
        ]
    lookup = {
        (
            int(row["realization"]),
            round(float(row["parameter__Gamma0"]), 12),
        ): row
        for row in rows
    }
    if len(lookup) != seeds.size * gamma.size:
        raise ValueError(f"incomplete landscape grid in {run}")

    reference_grid: np.ndarray | None = None
    derivatives: np.ndarray | None = None
    maximum_death = max(float(row["death_time_maximum"]) for row in rows)

    for seed_index, seed in enumerate(seeds):
        values_by_gamma: list[np.ndarray] = []
        for gamma_value in gamma:
            row = lookup[(int(seed), round(float(gamma_value), 12))]
            path = run / "landscapes" / f"{row['matrix_id']}.npz"
            with np.load(path, allow_pickle=False) as archive:
                grid = np.asarray(archive["H0_grid"], dtype=float)
                values = np.asarray(archive["H0_values"], dtype=np.float32)
            if values.shape != (L - 1, grid.size):
                raise ValueError(f"unexpected landscape shape in {path}")
            if reference_grid is None:
                reference_grid = grid
                derivatives = np.empty(
                    (
                        gamma.size,
                        seeds.size,
                        (L - 1) * grid.size,
                    ),
                    dtype=np.float32,
                )
            elif not np.array_equal(grid, reference_grid):
                raise ValueError("individual landscape grids differ")
            values_by_gamma.append(values)

        tensor = np.stack(values_by_gamma).astype(np.float64)
        derivative = np.gradient(tensor, gamma, axis=0, edge_order=2)
        assert reference_grid is not None and derivatives is not None
        steps = np.diff(reference_grid)
        if not np.allclose(steps, steps[0], rtol=1e-12, atol=1e-15):
            raise ValueError("the landscape grid is not uniform")
        weights = np.full(reference_grid.size, float(steps[0]))
        weights[[0, -1]] *= 0.5
        weighted = derivative * np.sqrt(weights[None, None, :] / (L - 1))
        derivatives[:, seed_index] = weighted.reshape(
            gamma.size,
            -1,
        ).astype(np.float32)

    assert derivatives is not None
    gram = np.empty(
        (gamma.size, seeds.size, seeds.size),
        dtype=np.float64,
    )
    for gamma_index, current in enumerate(derivatives):
        gram[gamma_index] = np.asarray(current @ current.T, dtype=np.float64)
    return gram, float(maximum_death)


def bootstrap_sample_statistics(
    peaks: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    samples = peaks[indices]
    return np.mean(samples, axis=1), np.std(samples, axis=1, ddof=1)


def main() -> None:
    arguments = parse_args()
    if arguments.bootstraps < 100:
        raise ValueError("use at least 100 bootstrap replicates")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = arguments.output_directory.expanduser().resolve()
    sentinel = output / "summary.json"
    if sentinel.exists() and not arguments.overwrite:
        raise FileExistsError(f"{sentinel} exists; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    package_manifest_path = FIGURE_DATA / "source_manifest.json"
    package_manifest = json.loads(
        package_manifest_path.read_text(encoding="utf-8")
    )
    sizes = np.asarray(
        sorted(int(value) for value in package_manifest["sizes"]),
        dtype=int,
    )
    rng = np.random.default_rng(arguments.rng_seed)
    bootstrap_count = int(arguments.bootstraps)

    ensemble_physical_peak = np.empty(sizes.size)
    ensemble_landscape_peak = np.empty(sizes.size)
    ensemble_physical_bootstrap = np.empty((bootstrap_count, sizes.size))
    ensemble_landscape_bootstrap = np.empty_like(
        ensemble_physical_bootstrap
    )
    sample_physical_mean = np.empty(sizes.size)
    sample_landscape_mean = np.empty(sizes.size)
    sample_physical_width = np.empty(sizes.size)
    sample_landscape_width = np.empty(sizes.size)
    sample_physical_mean_bootstrap = np.empty_like(
        ensemble_physical_bootstrap
    )
    sample_landscape_mean_bootstrap = np.empty_like(
        ensemble_physical_bootstrap
    )
    sample_physical_width_bootstrap = np.empty_like(
        ensemble_physical_bootstrap
    )
    sample_landscape_width_bootstrap = np.empty_like(
        ensemble_physical_bootstrap
    )
    saved_arrays: dict[str, np.ndarray] = {"L_values": sizes}
    size_summaries: dict[str, Any] = {}
    source_runs: dict[str, Any] = {}

    individual_rows: list[dict[str, Any]] = []
    scaling_rows: list[dict[str, Any]] = []

    for size_index, L_value in enumerate(sizes):
        L = int(L_value)
        entry = package_manifest["sizes"][str(L)]
        chi = int(entry["chi_max"])
        physical_path = (
            FIGURE_DATA
            / f"random_h0_L{L}_chi{chi}_ensemble_arrays.npz"
        )
        topology_path = (
            FIGURE_DATA
            / f"random_h0_L{L}_chi{chi}_topology_landscapes.npz"
        )
        with np.load(physical_path, allow_pickle=False) as archive:
            gamma = np.asarray(archive["Gamma0"], dtype=float)
            seeds = np.asarray(archive["seeds"], dtype=int)
            bulk = np.asarray(archive["bulk_ms2_by_seed"], dtype=float)
        if bulk.shape != (seeds.size, gamma.size):
            if bulk.T.shape == (seeds.size, gamma.size):
                bulk = bulk.T
            else:
                raise ValueError(f"unexpected bulk array shape for L={L}")
        with np.load(topology_path, allow_pickle=False) as archive:
            landscape_grid = np.asarray(
                archive["landscape_grid"],
                dtype=float,
            )
            mean_landscape = np.asarray(
                archive["mean_landscape"],
                dtype=float,
            )

        indices = rng.integers(
            0,
            seeds.size,
            size=(bootstrap_count, seeds.size),
            endpoint=False,
        )
        counts = np.asarray(
            [
                np.bincount(row, minlength=seeds.size)
                for row in indices
            ],
            dtype=float,
        )
        resampling_weights = counts / seeds.size

        physical_response = -np.gradient(
            bulk,
            gamma,
            axis=1,
            edge_order=2,
        )
        physical_peak_details = local_quadratic_peak(
            gamma,
            np.mean(physical_response, axis=0),
        )
        ensemble_physical_peak[size_index] = float(
            physical_peak_details["Gamma0"]
        )
        resampled_physical_response = (
            resampling_weights @ physical_response
        )
        ensemble_physical_bootstrap[:, size_index] = peaks_for_curves(
            gamma,
            resampled_physical_response,
        )[0]
        (
            physical_sample_peaks,
            physical_discrete_peaks,
            physical_edge_flags,
            physical_quadratic_flags,
        ) = peaks_for_curves(gamma, physical_response)

        run = ROOT / entry["tda_run"]["directory"]
        gram, maximum_finite_death = landscape_gram_matrices(
            run,
            gamma,
            seeds,
            L,
        )
        equal_weights = np.full(seeds.size, 1.0 / seeds.size)
        mean_velocity_from_gram = np.sqrt(
            np.maximum(
                np.asarray(
                    [
                        equal_weights @ current @ equal_weights
                        for current in gram
                    ]
                ),
                0.0,
            )
        )
        mean_landscape_derivative = np.gradient(
            mean_landscape,
            gamma,
            axis=0,
            edge_order=2,
        )
        mean_velocity_direct = np.sqrt(
            np.trapezoid(
                np.sum(mean_landscape_derivative**2, axis=1),
                landscape_grid,
                axis=1,
            )
            / (L - 1)
        )
        gram_error = float(
            np.max(np.abs(mean_velocity_from_gram - mean_velocity_direct))
        )
        if gram_error > 2.0e-6:
            raise ValueError(
                f"landscape Gram reconstruction failed for L={L}: "
                f"{gram_error}"
            )
        landscape_peak_details = local_quadratic_peak(
            gamma,
            mean_velocity_direct,
        )
        ensemble_landscape_peak[size_index] = float(
            landscape_peak_details["Gamma0"]
        )

        velocity_squared = np.empty(
            (bootstrap_count, gamma.size),
            dtype=float,
        )
        for gamma_index, current in enumerate(gram):
            velocity_squared[:, gamma_index] = np.sum(
                (resampling_weights @ current) * resampling_weights,
                axis=1,
            )
        resampled_landscape_velocity = np.sqrt(
            np.maximum(velocity_squared, 0.0)
        )
        ensemble_landscape_bootstrap[:, size_index] = peaks_for_curves(
            gamma,
            resampled_landscape_velocity,
        )[0]

        individual_landscape_velocity = np.sqrt(
            np.maximum(
                np.diagonal(gram, axis1=1, axis2=2).T,
                0.0,
            )
        )
        (
            landscape_sample_peaks,
            landscape_discrete_peaks,
            landscape_edge_flags,
            landscape_quadratic_flags,
        ) = peaks_for_curves(gamma, individual_landscape_velocity)

        sample_physical_mean[size_index] = float(
            np.mean(physical_sample_peaks)
        )
        sample_landscape_mean[size_index] = float(
            np.mean(landscape_sample_peaks)
        )
        sample_physical_width[size_index] = float(
            np.std(physical_sample_peaks, ddof=1)
        )
        sample_landscape_width[size_index] = float(
            np.std(landscape_sample_peaks, ddof=1)
        )
        (
            sample_physical_mean_bootstrap[:, size_index],
            sample_physical_width_bootstrap[:, size_index],
        ) = bootstrap_sample_statistics(physical_sample_peaks, indices)
        (
            sample_landscape_mean_bootstrap[:, size_index],
            sample_landscape_width_bootstrap[:, size_index],
        ) = bootstrap_sample_statistics(landscape_sample_peaks, indices)

        saved_arrays.update(
            {
                f"L{L}_Gamma0": gamma,
                f"L{L}_seeds": seeds,
                f"L{L}_physical_response_by_seed": physical_response,
                f"L{L}_physical_Gamma0_star_by_seed": physical_sample_peaks,
                f"L{L}_landscape_velocity_by_seed": (
                    individual_landscape_velocity
                ),
                f"L{L}_landscape_Gamma0_star_by_seed": (
                    landscape_sample_peaks
                ),
            }
        )

        for observable, peaks, discrete, edge, quadratic in (
            (
                "physical",
                physical_sample_peaks,
                physical_discrete_peaks,
                physical_edge_flags,
                physical_quadratic_flags,
            ),
            (
                "landscape_velocity",
                landscape_sample_peaks,
                landscape_discrete_peaks,
                landscape_edge_flags,
                landscape_quadratic_flags,
            ),
        ):
            for seed, peak, discrete_peak, edge_flag, quadratic_flag in zip(
                seeds,
                peaks,
                discrete,
                edge,
                quadratic,
            ):
                individual_rows.append(
                    {
                        "observable": observable,
                        "L": L,
                        "chi_max": chi,
                        "seed": int(seed),
                        "Gamma0_star": float(peak),
                        "discrete_Gamma0_star": float(discrete_peak),
                        "edge_maximum": int(edge_flag),
                        "used_quadratic": int(quadratic_flag),
                    }
                )

        size_summaries[str(L)] = {
            "chi_max": chi,
            "seed_count": int(seeds.size),
            "Gamma0_grid": {
                "minimum": float(gamma[0]),
                "maximum": float(gamma[-1]),
                "points": int(gamma.size),
                "minimum_step": float(np.min(np.diff(gamma))),
            },
            "maximum_finite_H0_death": maximum_finite_death,
            "mean_landscape_Gram_max_abs_error": gram_error,
            "physical": {
                "ensemble_curve_peak": physical_peak_details,
                "individual_peak_mean": sample_physical_mean[size_index],
                "individual_peak_std": sample_physical_width[size_index],
                "individual_edge_count": int(
                    np.count_nonzero(physical_edge_flags)
                ),
                "individual_quadratic_count": int(
                    np.count_nonzero(physical_quadratic_flags)
                ),
            },
            "landscape_velocity": {
                "ensemble_curve_peak": landscape_peak_details,
                "individual_peak_mean": sample_landscape_mean[size_index],
                "individual_peak_std": sample_landscape_width[size_index],
                "individual_edge_count": int(
                    np.count_nonzero(landscape_edge_flags)
                ),
                "individual_quadratic_count": int(
                    np.count_nonzero(landscape_quadratic_flags)
                ),
            },
        }
        source_runs[str(L)] = {
            "physical_file": str(physical_path.relative_to(ROOT).as_posix()),
            "physical_sha256": sha256_file(physical_path),
            "topology_file": str(topology_path.relative_to(ROOT).as_posix()),
            "topology_sha256": sha256_file(topology_path),
            "tda_run": str(run.relative_to(ROOT).as_posix()),
            "summaries_sha256": sha256_file(run / "summaries.csv"),
        }
        print(
            f"L={L}: ensemble peaks "
            f"physical={ensemble_physical_peak[size_index]:.6f}, "
            f"landscape={ensemble_landscape_peak[size_index]:.6f}; "
            f"sample means={sample_physical_mean[size_index]:.6f},"
            f"{sample_landscape_mean[size_index]:.6f}; "
            f"sample widths={sample_physical_width[size_index]:.6f},"
            f"{sample_landscape_width[size_index]:.6f}; "
            f"edge counts={np.count_nonzero(physical_edge_flags)},"
            f"{np.count_nonzero(landscape_edge_flags)}"
        )

    sequences = {
        "ensemble_curve_shift_physical": np.abs(
            ensemble_physical_peak - EXACT_CRITICAL_GAMMA0
        ),
        "ensemble_curve_shift_landscape_velocity": np.abs(
            ensemble_landscape_peak - EXACT_CRITICAL_GAMMA0
        ),
        "mean_individual_peak_shift_physical": np.abs(
            sample_physical_mean - EXACT_CRITICAL_GAMMA0
        ),
        "mean_individual_peak_shift_landscape_velocity": np.abs(
            sample_landscape_mean - EXACT_CRITICAL_GAMMA0
        ),
        "individual_peak_width_physical": sample_physical_width,
        "individual_peak_width_landscape_velocity": sample_landscape_width,
    }
    bootstrap_sequences = {
        "ensemble_curve_shift_physical": np.abs(
            ensemble_physical_bootstrap - EXACT_CRITICAL_GAMMA0
        ),
        "ensemble_curve_shift_landscape_velocity": np.abs(
            ensemble_landscape_bootstrap - EXACT_CRITICAL_GAMMA0
        ),
        "mean_individual_peak_shift_physical": np.abs(
            sample_physical_mean_bootstrap - EXACT_CRITICAL_GAMMA0
        ),
        "mean_individual_peak_shift_landscape_velocity": np.abs(
            sample_landscape_mean_bootstrap - EXACT_CRITICAL_GAMMA0
        ),
        "individual_peak_width_physical": (
            sample_physical_width_bootstrap
        ),
        "individual_peak_width_landscape_velocity": (
            sample_landscape_width_bootstrap
        ),
    }

    fit_rows: list[dict[str, Any]] = []
    fits: dict[str, Any] = {}
    for sequence_name, sequence_values in sequences.items():
        fits[sequence_name] = {}
        for minimum_size in FIT_MINIMA:
            point_fit = power_law_fit(
                sizes,
                sequence_values,
                minimum_size=minimum_size,
            )
            bootstrap_exponents = bootstrap_power_exponents(
                sizes,
                bootstrap_sequences[sequence_name],
                minimum_size=minimum_size,
            )
            exponent_summary = finite_summary(bootstrap_exponents)
            point_fit["p_bootstrap"] = exponent_summary
            fits[sequence_name][f"L_ge_{minimum_size}"] = point_fit
            fit_rows.append(
                {
                    "sequence": sequence_name,
                    "minimum_size": minimum_size,
                    "sizes": ";".join(
                        str(value) for value in point_fit["sizes"]
                    ),
                    "A": point_fit["A"],
                    "p": point_fit["p"],
                    "p_bootstrap_q025": exponent_summary["q025"],
                    "p_bootstrap_median": exponent_summary["median"],
                    "p_bootstrap_q975": exponent_summary["q975"],
                    "log_rmse": point_fit["log_rmse"],
                    "log_r2": point_fit["log_r2"],
                }
            )

    for index, L in enumerate(sizes):
        row: dict[str, Any] = {
            "L": int(L),
            "seed_count": size_summaries[str(int(L))]["seed_count"],
            "physical_ensemble_Gamma0_star": ensemble_physical_peak[index],
            "landscape_ensemble_Gamma0_star": ensemble_landscape_peak[index],
            "physical_individual_Gamma0_star_mean": sample_physical_mean[index],
            "landscape_individual_Gamma0_star_mean": sample_landscape_mean[index],
            "physical_individual_Gamma0_star_std": sample_physical_width[index],
            "landscape_individual_Gamma0_star_std": sample_landscape_width[index],
        }
        for name, values in sequences.items():
            row[name] = float(values[index])
        scaling_rows.append(row)

    summary = {
        "schema_version": 1,
        "analysis": (
            "IRFP-motivated random h=0 pseudocritical shift and "
            "distribution-width scaling"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary_conditions": "OBC",
        "ensemble": "fixed_mean",
        "exact_critical_Gamma0": EXACT_CRITICAL_GAMMA0,
        "matrix_convention": "C=Cstag; topology uses D=1-Cstag",
        "bootstrap_count": bootstrap_count,
        "rng_seed": int(arguments.rng_seed),
        "fit_model": "y(L)=A L^{-p}, fitted in log space",
        "fit_interpretation": {
            "shift_reference": "p=1/nu_typ=1",
            "width_reference": "p=1/nu=1/2",
            "caveat": (
                "fixed_mean imposes an arithmetic-mean constraint and is "
                "related to, but not identical with, the logarithmic "
                "microcanonical constraint studied by Monthus"
            ),
        },
        "definitions": {
            "ensemble_curve_shift": (
                "absolute difference between the peak of the disorder-mean "
                "response curve and Gamma0_c=1"
            ),
            "mean_individual_peak_shift": (
                "absolute difference between the arithmetic mean of the "
                "per-seed peak positions and Gamma0_c=1"
            ),
            "individual_peak_width": (
                "sample standard deviation of per-seed peak positions"
            ),
            "peak_refinement": (
                "local quadratic through the discrete maximum and its two "
                "neighbours; edge maxima remain discrete and are flagged"
            ),
        },
        "sizes": size_summaries,
        "sequences": {
            name: {
                str(int(L)): float(value)
                for L, value in zip(sizes, values)
            }
            for name, values in sequences.items()
        },
        "fits": fits,
        "sources": {
            "figure_source_manifest": {
                "path": str(package_manifest_path.relative_to(ROOT).as_posix()),
                "sha256": sha256_file(package_manifest_path),
            },
            "sizes": source_runs,
        },
    }

    np.savez_compressed(
        output / "random_h0_irfp_pseudocritical_scaling.npz",
        **saved_arrays,
        ensemble_physical_Gamma0_star=ensemble_physical_peak,
        ensemble_landscape_Gamma0_star=ensemble_landscape_peak,
        ensemble_physical_bootstrap=ensemble_physical_bootstrap,
        ensemble_landscape_bootstrap=ensemble_landscape_bootstrap,
        sample_physical_mean_bootstrap=sample_physical_mean_bootstrap,
        sample_landscape_mean_bootstrap=sample_landscape_mean_bootstrap,
        sample_physical_width_bootstrap=sample_physical_width_bootstrap,
        sample_landscape_width_bootstrap=sample_landscape_width_bootstrap,
        **{
            f"sequence_{name}": values
            for name, values in sequences.items()
        },
        **{
            f"bootstrap_sequence_{name}": values
            for name, values in bootstrap_sequences.items()
        },
    )
    sentinel.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (output / "individual_pseudocritical_points.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(individual_rows[0]),
        )
        writer.writeheader()
        writer.writerows(individual_rows)
    with (output / "scaling_sequences.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scaling_rows[0]))
        writer.writeheader()
        writer.writerows(scaling_rows)
    with (output / "power_law_fits.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fit_rows[0]))
        writer.writeheader()
        writer.writerows(fit_rows)

    colors = {
        "physical": "#0072B2",
        "landscape": "#D55E00",
    }
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.0),
        constrained_layout=True,
    )
    plot_sizes = np.geomspace(float(sizes.min()), float(sizes.max()), 300)

    for axis, physical_values, landscape_values, kind in (
        (
            axes[0],
            sequences["mean_individual_peak_shift_physical"],
            sequences["mean_individual_peak_shift_landscape_velocity"],
            "shift",
        ),
        (
            axes[1],
            sequences["individual_peak_width_physical"],
            sequences["individual_peak_width_landscape_velocity"],
            "width",
        ),
    ):
        for values, name, color, marker in (
            (physical_values, "physical", colors["physical"], "o"),
            (
                landscape_values,
                "landscape",
                colors["landscape"],
                "s",
            ),
        ):
            sequence_suffix = (
                "landscape_velocity" if name == "landscape" else name
            )
            sequence_name = (
                f"mean_individual_peak_shift_{sequence_suffix}"
                if kind == "shift"
                else f"individual_peak_width_{sequence_suffix}"
            )
            bootstrap_values = bootstrap_sequences[sequence_name]
            low, high = np.quantile(
                bootstrap_values,
                (0.025, 0.975),
                axis=0,
            )
            axis.errorbar(
                sizes,
                values,
                yerr=np.vstack((values - low, high - values)),
                fmt=marker,
                color=color,
                capsize=2.5,
                label=name.replace("_", " "),
            )
            fit = fits[sequence_name]["L_ge_16"]
            axis.plot(
                plot_sizes,
                fit["A"] * plot_sizes ** (-fit["p"]),
                color=color,
                linestyle="-" if name == "physical" else "--",
                label=fr"{name}: $p={fit['p']:.2f}$",
            )

        reference_exponent = 1.0 if kind == "shift" else 0.5
        reference_values = physical_values
        reference_anchor = float(
            np.exp(
                np.mean(
                    np.log(reference_values[sizes >= 16])
                    + reference_exponent * np.log(sizes[sizes >= 16])
                )
            )
        )
        axis.plot(
            plot_sizes,
            reference_anchor * plot_sizes ** (-reference_exponent),
            color="black",
            linestyle=":",
            label=fr"reference $L^{{-{reference_exponent:g}}}$",
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel(r"$L$")
        axis.grid(alpha=0.2, which="both")
        axis.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel(
        r"$|\overline{\Gamma_{0,\alpha}^{\star}}-1|$"
    )
    axes[0].set_title("Shift of mean individual pseudocritical point")
    axes[1].set_ylabel(
        r"$\sigma(\Gamma_{0,\alpha}^{\star})$"
    )
    axes[1].set_title("Width of pseudocritical distribution")
    for suffix in ("png", "pdf"):
        figure.savefig(
            output
            / f"figure_random_h0_individual_pseudocritical_scaling.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)

    print(f"Analysis written to: {output}")
    for row in fit_rows:
        if row["minimum_size"] == 16:
            print(
                f"{row['sequence']}: p={float(row['p']):.6f}, "
                f"95% bootstrap "
                f"[{float(row['p_bootstrap_q025']):.6f}, "
                f"{float(row['p_bootstrap_q975']):.6f}]"
            )


if __name__ == "__main__":
    main()
