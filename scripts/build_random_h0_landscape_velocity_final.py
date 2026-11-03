"""Build the closed random-h=0 physical/H0-landscape scaling package.

The script writes derived, reproducible products under results/. Registration
in paper_figures is deliberately handled by a separate explicit script.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

from analyze_random_h0_irfp_pseudocritical_scaling import (
    bootstrap_power_exponents,
    finite_summary,
    local_quadratic_peak,
    power_law_fit,
)


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results" / "random_ising_gamma_ensemble" / "random-h0-irfp-pseudocritical-scaling-v1" / "analysis" / "random_h0_irfp_pseudocritical_scaling.npz"
PHYSICAL = ROOT / "results" / "random_ising_gamma_ensemble" / "random-h0-L128-fine625-partial-physical-scaling-v1" / "analysis" / "random_h0_L128_partial_fine_physical_scaling.npz"
TOPOLOGY = ROOT / "results" / "random_ising_gamma_ensemble" / "random-h0-L128-fine625-partial-topology-scaling-v1" / "analysis" / "partial_fine_topology_scaling.npz"
OUTPUT = ROOT / "results" / "random_ising_gamma_ensemble" / "random-h0-landscape-velocity-final-fine50-v1" / "analysis"
SIZES = np.asarray([8, 16, 32, 64, 128], dtype=int)
CRITICAL = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_power_fit(sizes: np.ndarray, values: np.ndarray, p: float) -> dict[str, float]:
    mask = (sizes >= 16) & np.isfinite(values) & (values > 0)
    x = np.log(sizes[mask].astype(float))
    y = np.log(values[mask])
    intercept = float(np.mean(y + p * x))
    residual = y - (intercept - p * x)
    return {"A": float(np.exp(intercept)), "p": float(p), "log_rmse": float(np.sqrt(np.mean(residual**2)))}


def mean_landscape_velocity(L: int) -> tuple[np.ndarray, np.ndarray]:
    path = next((ROOT / "paper_figures" / "fig_random_h0" / "data_raw").glob(f"random_h0_L{L}_chi*_topology_landscapes.npz"))
    with np.load(path, allow_pickle=False) as archive:
        gamma = np.asarray(archive["Gamma0"], dtype=float)
        grid = np.asarray(archive["landscape_grid"], dtype=float)
        landscape = np.asarray(archive["mean_landscape"], dtype=float)
    derivative = np.gradient(landscape, gamma, axis=0, edge_order=2)
    velocity = np.sqrt(np.trapezoid(np.sum(derivative**2, axis=1), grid, axis=1) / (L - 1))
    return gamma, velocity


def fit_bundle(values: np.ndarray, bootstrap: np.ndarray, reference: float) -> dict:
    free = power_law_fit(SIZES, values, minimum_size=16)
    exponents = bootstrap_power_exponents(SIZES, bootstrap, minimum_size=16)
    free["p_bootstrap"] = finite_summary(exponents[np.isfinite(exponents)])
    free["reference_p"] = reference
    free["reference_in_bootstrap_95"] = bool(
        free["p_bootstrap"]["q025"] <= reference <= free["p_bootstrap"]["q975"]
    )
    return {"free_power": free, "fixed_reference_power": fixed_power_fit(SIZES, values, reference)}


def inverse_size_linear_fit(values: np.ndarray, bootstrap: np.ndarray) -> dict:
    mask = SIZES >= 16
    inverse_sizes = 1.0 / SIZES[mask].astype(float)
    slope, intercept = np.polyfit(inverse_sizes, values[mask], 1)
    bootstrap_coefficients = np.asarray(
        [np.polyfit(inverse_sizes, row[mask], 1) for row in bootstrap],
        dtype=float,
    )
    return {
        "fit_range": "L>=16",
        "slope": float(slope),
        "Gamma0_infinity": float(intercept),
        "Gamma0_infinity_bootstrap": finite_summary(bootstrap_coefficients[:, 1]),
    }


def main() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(OLD, allow_pickle=False) as archive:
        old = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(PHYSICAL, allow_pickle=False) as archive:
        physical = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(TOPOLOGY, allow_pickle=False) as archive:
        topology = {name: np.asarray(archive[name]) for name in archive.files}

    saved: dict[str, np.ndarray] = {"L_values": SIZES}
    ensemble_physical = np.empty(SIZES.size)
    ensemble_topology = np.empty(SIZES.size)
    individual_rows: list[dict] = []
    curves: dict[int, dict[str, np.ndarray]] = {}

    for index, L0 in enumerate(SIZES):
        L = int(L0)
        if L <= 32:
            gamma = old[f"L{L}_Gamma0"].astype(float)
            seeds = old[f"L{L}_seeds"].astype(int)
            response = old[f"L{L}_physical_response_by_seed"].astype(float)
            physical_peaks = old[f"L{L}_physical_Gamma0_star_by_seed"].astype(float)
            velocity_by_seed = old[f"L{L}_landscape_velocity_by_seed"].astype(float)
            topology_peaks = old[f"L{L}_landscape_Gamma0_star_by_seed"].astype(float)
            gamma_topology, velocity = mean_landscape_velocity(L)
            if not np.array_equal(gamma, gamma_topology):
                raise ValueError(f"physical/topological Gamma0 mismatch at L={L}")
        else:
            gamma = physical[f"L{L}_Gamma0"].astype(float)
            seeds = physical[f"L{L}_seeds"].astype(int)
            response = physical[f"L{L}_response_by_seed"].astype(float)
            physical_peaks = physical[f"L{L}_Gamma0_star_by_seed"].astype(float)
            gamma_topology = topology[f"L{L}_Gamma0"].astype(float)
            if not np.array_equal(gamma, gamma_topology):
                raise ValueError(f"fine physical/topological Gamma0 mismatch at L={L}")
            velocity_by_seed = topology[f"L{L}_landscape_velocity_by_seed"].astype(float)
            topology_peaks = topology[f"L{L}_Gamma0_star_by_seed"].astype(float)
            velocity = topology[f"L{L}_ensemble_landscape_velocity"].astype(float)
        if seeds.size != 50:
            raise ValueError(f"final closure requires 50 seeds at L={L}, found {seeds.size}")

        mean_response = np.mean(response, axis=0)
        ensemble_physical[index] = float(local_quadratic_peak(gamma, mean_response)["Gamma0"])
        ensemble_topology[index] = float(local_quadratic_peak(gamma, velocity)["Gamma0"])
        curves[L] = {"gamma": gamma, "response": mean_response, "velocity": velocity}
        saved.update({
            f"L{L}_Gamma0": gamma,
            f"L{L}_seeds": seeds,
            f"L{L}_physical_response_by_seed": response,
            f"L{L}_physical_Gamma0_star_by_seed": physical_peaks,
            f"L{L}_landscape_velocity_by_seed": velocity_by_seed,
            f"L{L}_landscape_Gamma0_star_by_seed": topology_peaks,
            f"L{L}_ensemble_physical_response": mean_response,
            f"L{L}_ensemble_landscape_velocity": velocity,
        })
        for seed, pp, tp in zip(seeds, physical_peaks, topology_peaks):
            individual_rows.append({"L": L, "seed": int(seed), "physical_Gamma0_star": float(pp), "landscape_Gamma0_star": float(tp)})

    raw_bootstrap = {
        "ensemble_physical": old["ensemble_physical_bootstrap"].astype(float).copy(),
        "ensemble_topology": old["ensemble_landscape_bootstrap"].astype(float).copy(),
        "mean_physical": old["sample_physical_mean_bootstrap"].astype(float).copy(),
        "mean_topology": old["sample_landscape_mean_bootstrap"].astype(float).copy(),
        "width_physical": old["sample_physical_width_bootstrap"].astype(float).copy(),
        "width_topology": old["sample_landscape_width_bootstrap"].astype(float).copy(),
    }
    for L in (64, 128):
        column = int(np.flatnonzero(SIZES == L)[0])
        raw_bootstrap["ensemble_physical"][:, column] = physical[f"L{L}_ensemble_Gamma0_star_bootstrap"]
        raw_bootstrap["mean_physical"][:, column] = physical[f"L{L}_mean_individual_Gamma0_star_bootstrap"]
        raw_bootstrap["width_physical"][:, column] = physical[f"L{L}_individual_width_bootstrap"]
        raw_bootstrap["ensemble_topology"][:, column] = topology[f"L{L}_ensemble_Gamma0_star_bootstrap"]
        raw_bootstrap["mean_topology"][:, column] = topology[f"L{L}_mean_individual_Gamma0_star_bootstrap"]
        raw_bootstrap["width_topology"][:, column] = topology[f"L{L}_individual_width_bootstrap"]

    mean_physical = np.asarray([np.mean(saved[f"L{L}_physical_Gamma0_star_by_seed"]) for L in SIZES])
    mean_topology = np.asarray([np.mean(saved[f"L{L}_landscape_Gamma0_star_by_seed"]) for L in SIZES])
    width_physical = np.asarray([np.std(saved[f"L{L}_physical_Gamma0_star_by_seed"], ddof=1) for L in SIZES])
    width_topology = np.asarray([np.std(saved[f"L{L}_landscape_Gamma0_star_by_seed"], ddof=1) for L in SIZES])
    sequences = {
        "ensemble_curve_shift_physical": np.abs(ensemble_physical - CRITICAL),
        "ensemble_curve_shift_landscape_velocity": np.abs(ensemble_topology - CRITICAL),
        "mean_individual_peak_shift_physical": np.abs(mean_physical - CRITICAL),
        "mean_individual_peak_shift_landscape_velocity": np.abs(mean_topology - CRITICAL),
        "individual_peak_width_physical": width_physical,
        "individual_peak_width_landscape_velocity": width_topology,
    }
    boot_sequences = {
        "ensemble_curve_shift_physical": np.abs(raw_bootstrap["ensemble_physical"] - CRITICAL),
        "ensemble_curve_shift_landscape_velocity": np.abs(raw_bootstrap["ensemble_topology"] - CRITICAL),
        "mean_individual_peak_shift_physical": np.abs(raw_bootstrap["mean_physical"] - CRITICAL),
        "mean_individual_peak_shift_landscape_velocity": np.abs(raw_bootstrap["mean_topology"] - CRITICAL),
        "individual_peak_width_physical": raw_bootstrap["width_physical"],
        "individual_peak_width_landscape_velocity": raw_bootstrap["width_topology"],
    }
    fits = {
        name: fit_bundle(values, boot_sequences[name], 0.5 if "width" in name else 1.0)
        for name, values in sequences.items()
    }
    inverse_size_fits = {
        "physical": inverse_size_linear_fit(ensemble_physical, raw_bootstrap["ensemble_physical"]),
        "landscape_velocity_H0": inverse_size_linear_fit(ensemble_topology, raw_bootstrap["ensemble_topology"]),
    }

    saved.update({
        "ensemble_physical_Gamma0_star": ensemble_physical,
        "ensemble_landscape_Gamma0_star": ensemble_topology,
        "ensemble_physical_bootstrap": raw_bootstrap["ensemble_physical"],
        "ensemble_landscape_bootstrap": raw_bootstrap["ensemble_topology"],
        "sample_physical_mean_bootstrap": raw_bootstrap["mean_physical"],
        "sample_landscape_mean_bootstrap": raw_bootstrap["mean_topology"],
        "sample_physical_width_bootstrap": raw_bootstrap["width_physical"],
        "sample_landscape_width_bootstrap": raw_bootstrap["width_topology"],
    })
    saved.update({f"sequence_{name}": value for name, value in sequences.items()})
    saved.update({f"bootstrap_sequence_{name}": value for name, value in boot_sequences.items()})
    np.savez_compressed(OUTPUT / "random_h0_irfp_pseudocritical_scaling.npz", **saved)

    with (OUTPUT / "individual_pseudocritical_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(individual_rows[0]))
        writer.writeheader(); writer.writerows(individual_rows)
    sequence_rows = []
    for index, L in enumerate(SIZES):
        row = {"L": int(L), "seed_count": 50, "physical_ensemble_Gamma0_star": ensemble_physical[index], "landscape_ensemble_Gamma0_star": ensemble_topology[index], "physical_individual_Gamma0_star_mean": mean_physical[index], "landscape_individual_Gamma0_star_mean": mean_topology[index], "physical_individual_Gamma0_star_std": width_physical[index], "landscape_individual_Gamma0_star_std": width_topology[index]}
        row.update({name: values[index] for name, values in sequences.items()})
        sequence_rows.append(row)
    with (OUTPUT / "scaling_sequences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sequence_rows[0])); writer.writeheader(); writer.writerows(sequence_rows)
    fit_rows = []
    for name, bundle in fits.items():
        free = bundle["free_power"]
        fit_rows.append({"sequence": name, "minimum_size": 16, "sizes": ";".join(map(str, free["sizes"])), "A": free["A"], "p": free["p"], "p_bootstrap_q025": free["p_bootstrap"]["q025"], "p_bootstrap_median": free["p_bootstrap"]["median"], "p_bootstrap_q975": free["p_bootstrap"]["q975"], "reference_p": free["reference_p"], "reference_in_bootstrap_95": int(free["reference_in_bootstrap_95"]), "log_rmse": free["log_rmse"], "fixed_reference_log_rmse": bundle["fixed_reference_power"]["log_rmse"]})
    with (OUTPUT / "power_law_fits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fit_rows[0])); writer.writeheader(); writer.writerows(fit_rows)
    inverse_rows = []
    for observable, result in inverse_size_fits.items():
        inverse_rows.append({
            "observable": observable,
            "fit_range": result["fit_range"],
            "slope": result["slope"],
            "Gamma0_infinity": result["Gamma0_infinity"],
            "Gamma0_infinity_bootstrap_q025": result["Gamma0_infinity_bootstrap"]["q025"],
            "Gamma0_infinity_bootstrap_median": result["Gamma0_infinity_bootstrap"]["median"],
            "Gamma0_infinity_bootstrap_q975": result["Gamma0_infinity_bootstrap"]["q975"],
        })
    with (OUTPUT / "inverse_size_linear_fits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inverse_rows[0])); writer.writeheader(); writer.writerows(inverse_rows)

    velocity_rows = []
    for L in SIZES:
        for gamma, response, velocity in zip(curves[int(L)]["gamma"], curves[int(L)]["response"], curves[int(L)]["velocity"]):
            velocity_rows.append({"L": int(L), "seed_count": 50, "Gamma0": gamma, "physical_response": response, "landscape_velocity": velocity})
    with (OUTPUT / "response_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(velocity_rows[0])); writer.writeheader(); writer.writerows(velocity_rows)

    colors = dict(zip(SIZES, ["#443983", "#31688e", "#35b779", "#90d743", "#fde725"]))
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.2), constrained_layout=True)
    for L in SIZES:
        L = int(L); c = colors[L]; gamma = curves[L]["gamma"]
        axes[0].plot(gamma, curves[L]["response"], "o-", ms=2.6, lw=1.5, color=c, label=fr"$L={L}$")
        axes[0].scatter(ensemble_physical[np.flatnonzero(SIZES == L)[0]], np.interp(ensemble_physical[np.flatnonzero(SIZES == L)[0]], gamma, curves[L]["response"]), s=62, color=c, edgecolor="black", zorder=4)
        axes[1].plot(gamma, curves[L]["velocity"], "o-", ms=2.6, lw=1.5, color=c, label=fr"$L={L}$")
        axes[1].scatter(ensemble_topology[np.flatnonzero(SIZES == L)[0]], np.interp(ensemble_topology[np.flatnonzero(SIZES == L)[0]], gamma, curves[L]["velocity"]), marker="s", s=58, color=c, edgecolor="black", zorder=4)
    for axis in axes[:2]:
        axis.axvline(1, color="black", ls="--", lw=1.2); axis.grid(alpha=.2); axis.set_xlabel(r"$\Gamma_0$"); axis.legend(frameon=False, fontsize=8, ncol=1); axis.tick_params(direction="in", top=True, right=True)
    axes[0].set_ylabel(r"$\langle-\partial_{\Gamma_0}m_{s,\mathrm{bulk}}^2\rangle$")
    axes[1].set_ylabel(r"$\xi_\lambda=\|\partial_{\Gamma_0}\langle\lambda_0\rangle\|_2/\sqrt{L-1}$")
    axes[0].text(.02, .96, "(a)", transform=axes[0].transAxes, va="top", fontweight="bold")
    axes[1].text(.02, .96, "(b)", transform=axes[1].transAxes, va="top", fontweight="bold")
    plot_L = np.geomspace(16, 128, 300)
    inverse_sizes = 1.0 / SIZES.astype(float)
    fit_x = np.linspace(0.0, 1.05 / 8.0, 300)
    for values, bs, fit_key, color, marker, linestyle, label in (
        (ensemble_physical, raw_bootstrap["ensemble_physical"], "physical", "#0072B2", "o", "-", "physical"),
        (ensemble_topology, raw_bootstrap["ensemble_topology"], "landscape_velocity_H0", "#D55E00", "s", "--", r"$H_0$ landscape velocity"),
    ):
        lo, hi = np.quantile(bs, [.025, .975], axis=0)
        filled = SIZES >= 16
        opened = ~filled
        axes[2].errorbar(inverse_sizes[filled], values[filled], yerr=np.vstack([values-lo, hi-values])[:, filled], color=color, marker=marker, lw=0, elinewidth=1.2, capsize=3, label=label)
        axes[2].errorbar(inverse_sizes[opened], values[opened], yerr=np.vstack([values-lo, hi-values])[:, opened], color=color, marker=marker, markerfacecolor="white", lw=0, elinewidth=1.2, capsize=3)
        result = inverse_size_fits[fit_key]
        axes[2].plot(fit_x, result["Gamma0_infinity"] + result["slope"] * fit_x, color=color, ls=linestyle, lw=1.8, label=fr"fit: $\Gamma_{{0,c}}={result['Gamma0_infinity']:.3f}$")
    axes[2].axhline(1.0, color="black", ls="--", lw=1.2, label=r"exact $\Gamma_{0,c}=1$")
    axes[2].set(xlabel=r"$1/L$", ylabel=r"$\Gamma_0^\star(L)$", xlim=(-.003, 1.05 / 8.0))
    axes[2].grid(alpha=.2); axes[2].legend(frameon=False, fontsize=7); axes[2].tick_params(direction="in", top=True, right=True); axes[2].text(.02, .96, "(c)", transform=axes[2].transAxes, va="top", fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"figure_random_h0_physics_landscape_velocity_3panel.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.1, 4.0), constrained_layout=True)
    panels = [("mean_individual_peak_shift", 1.0, r"$|\overline{\Gamma_0^\star}-1|$"), ("individual_peak_width", .5, r"$\sigma(\Gamma_0^\star)$")]
    for axis, (stem, reference, ylabel) in zip(axes, panels):
        for suffix, color, marker, label in (("physical", "#0072B2", "o", "physical"), ("landscape_velocity", "#D55E00", "s", r"$H_0$ landscape velocity")):
            name = f"{stem}_{suffix}"; values = sequences[name]; bs = boot_sequences[name]; lo, hi = np.quantile(bs, [.025, .975], axis=0); free = fits[name]["free_power"]
            axis.errorbar(SIZES, values, yerr=np.vstack([values-lo, hi-values]), color=color, marker=marker, lw=0, elinewidth=1.2, capsize=3, label=label)
            axis.plot(plot_L, free["A"] * plot_L**(-free["p"]), color=color, label=fr"fit: $p={free['p']:.2f}$")
        axis.set(xscale="log", yscale="log", xlabel=r"$L$", ylabel=ylabel); axis.set_xticks(SIZES, labels=[str(x) for x in SIZES]); axis.grid(alpha=.2, which="both"); axis.legend(frameon=False, fontsize=8); axis.tick_params(direction="in", top=True, right=True)
    axes[0].text(.02, .96, "(a)", transform=axes[0].transAxes, va="top", fontweight="bold"); axes[1].text(.02, .96, "(b)", transform=axes[1].transAxes, va="top", fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"figure_random_h0_pseudocritical_distribution_scaling_2panel.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

    source_paths = [OLD, PHYSICAL, TOPOLOGY]
    source_paths.extend(
        next((ROOT / "paper_figures" / "fig_random_h0" / "data_raw").glob(f"random_h0_L{L}_chi*_topology_landscapes.npz"))
        for L in (8, 16, 32)
    )
    summary = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "closed random-h=0 finite-size figure",
        "boundary_conditions": "OBC",
        "ensemble": "fixed_mean",
        "matrix_convention": "C=Cstag; persistence computed after D=1-C",
        "sizes": {str(int(L)): {"seed_count": 50, "Gamma0_points": int(curves[int(L)]["gamma"].size), "Gamma0_min": float(curves[int(L)]["gamma"][0]), "Gamma0_max": float(curves[int(L)]["gamma"][-1])} for L in SIZES},
        "fit_range": "L>=16; L=8 displayed but excluded",
        "fit_interpretation": {
            "main_figure": "empirical Gamma0_star = Gamma0_infinity + a/L; no L^-1 or L^-1/2 reference fits drawn",
            "caveat": "fixed_mean fixes the arithmetic rather than logarithmic sample mean; the accessible sizes remain pre-asymptotic",
        },
        "fits": fits,
        "inverse_size_linear_fits": inverse_size_fits,
        "sequences": {name: {str(int(L)): float(value) for L, value in zip(SIZES, values)} for name, values in sequences.items()},
        "sources": {str(path.relative_to(ROOT).as_posix()): {"sha256": sha256(path), "size_bytes": path.stat().st_size} for path in source_paths},
        "interpretation": "H0 landscape peak positions approach Gamma0c but remain pre-asymptotic; the H0 sample-to-sample width is compatible with L^-1/2 over L=16..128.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "fits": fits}, indent=2))


if __name__ == "__main__":
    main()
