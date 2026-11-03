"""Reanalyse the clean L=64 targeted boundary with H0/H1 observables.

This is an exploratory analysis.  It joins only the mutually compatible
OBC, L=64, chi=128 targeted campaigns, computes topology from C=Cstag
(using D=1-C internally), and keeps the problematic h=1.95 cut flagged.
No homology dimensions above H1 are computed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "clean_ising_gamma_campaign"
CAMPAIGNS = (
    "clean-fig1-bound-L64-c128-v1",
    "clean-fig1-refine-L64-c128-v1",
    "clean-fig1-refine2-L64-c128-v1",
)
DEFAULT_OUTPUT = BASE / "clean-L64-c128-new-topology-v1" / "analysis"
GRID = np.linspace(0.0, 2.0, 401)
LAYERS = 15
RANKED_K = 5


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_paths(campaign: Path) -> list[Path]:
    return sorted((campaign / "scans").glob("*/continuation_summary.npz"))


def validate_configs() -> list[Path]:
    configs = []
    config_paths = []
    for name in CAMPAIGNS:
        path = BASE / name / "campaign_config.json"
        config_paths.append(path)
        configs.append(json.loads(path.read_text(encoding="utf-8")))
    keys = (
        "model", "disorder", "J", "bc", "conserve", "chi_max",
        "initial_states", "two_site_sweeps", "one_site_sweeps",
        "max_E_err", "max_S_err", "svd_min", "mixer",
        "branch_energy_tolerance", "branch_infidelity_tolerance",
        "selection_energy_tolerance", "mps_save_policy",
    )
    for config in configs[1:]:
        differences = [key for key in keys if config.get(key) != configs[0].get(key)]
        if differences:
            raise ValueError("incompatible campaigns: " + ", ".join(differences))
    expected = {
        "model": "clean_mixed_field_ising", "disorder": "none",
        "bc": "open", "chi_max": 128,
    }
    for key, value in expected.items():
        if configs[0].get(key) != value:
            raise ValueError(f"unexpected {key}: {configs[0].get(key)!r}")
    return config_paths


def load_curves() -> tuple[list[dict[str, np.ndarray]], list[Path]]:
    by_h: dict[float, dict[float, dict[str, object]]] = {}
    sources: list[Path] = []
    target_by_h: dict[float, float] = {}
    base_config = json.loads((BASE / CAMPAIGNS[0] / "campaign_config.json").read_text(encoding="utf-8"))
    for point in base_config["Gamma_values_by_h"]:
        target_by_h[round(float(point["h"]), 12)] = float(point["Gamma_center"])

    for campaign_name in CAMPAIGNS:
        for path in scan_paths(BASE / campaign_name):
            sources.append(path)
            with np.load(path, allow_pickle=False) as archive:
                h = round(float(archive["h"]), 12)
                gamma = np.asarray(archive["Gamma_values"], dtype=float)
                cstag = np.asarray(archive["Cstag"], dtype=float)
                records = by_h.setdefault(h, {})
                for index, value in enumerate(gamma):
                    key = round(float(value), 12)
                    record = {
                        "Cstag": cstag[index],
                        "energy_density": float(archive["selected_energy_density"][index]),
                        "entropy": float(archive["half_chain_entropy"][index]),
                        "bond": float(archive["selected_max_bond_dimension"][index]),
                        "energy_difference": float(archive["absolute_energy_difference"][index]),
                        "branch_infidelity": float(max(0.0, 1.0 - archive["branch_fidelity"][index])),
                        "branch_disagreement": bool(archive["branch_disagreement"][index]),
                        "source": str(path.resolve()),
                    }
                    if key in records:
                        old = records[key]
                        if not np.allclose(old["Cstag"], record["Cstag"], rtol=1e-9, atol=1e-11):
                            raise ValueError(f"incompatible duplicate h={h:g}, Gamma={value:g}")
                    records[key] = record

    curves = []
    for h in sorted(by_h):
        points = by_h[h]
        gamma = np.asarray(sorted(points), dtype=float)
        cstag = np.stack([np.asarray(points[g]["Cstag"], dtype=float) for g in sorted(points)])
        L = cstag.shape[1]
        start, stop = L // 4, 3 * L // 4
        curves.append({
            "h": np.asarray(h),
            "target": np.asarray(target_by_h[h]),
            "Gamma": gamma,
            "Cstag": cstag,
            "bulk_ms2": np.mean(cstag[:, start:stop, start:stop], axis=(1, 2)),
            "paper_Cstag": np.mean(np.diagonal(cstag, offset=L // 2 - 1, axis1=1, axis2=2), axis=1),
            "energy_density": np.asarray([points[g]["energy_density"] for g in sorted(points)]),
            "entropy": np.asarray([points[g]["entropy"] for g in sorted(points)]),
            "bond": np.asarray([points[g]["bond"] for g in sorted(points)]),
            "energy_difference": np.asarray([points[g]["energy_difference"] for g in sorted(points)]),
            "branch_infidelity": np.asarray([points[g]["branch_infidelity"] for g in sorted(points)]),
            "branch_disagreement": np.asarray([points[g]["branch_disagreement"] for g in sorted(points)], dtype=bool),
        })
    return curves, sources


def mst_deaths(cstag: np.ndarray) -> np.ndarray:
    matrix = np.asarray(cstag, dtype=float)
    n = matrix.shape[0]
    distance = np.clip(1.0 - 0.5 * (matrix + matrix.T), 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    left, right = np.triu_indices(n, 1)
    order = np.argsort(distance[left, right], kind="stable")
    parent = np.arange(n)
    rank = np.zeros(n, dtype=np.int8)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    deaths = []
    for edge in order:
        a, b = find(int(left[edge])), find(int(right[edge]))
        if a == b:
            continue
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1
        deaths.append(float(distance[left[edge], right[edge]]))
        if len(deaths) == n - 1:
            break
    return np.sort(np.asarray(deaths, dtype=float))[::-1]


def h1_diagram(cstag: np.ndarray) -> np.ndarray:
    import gudhi

    matrix = np.asarray(cstag, dtype=float)
    distance = np.clip(1.0 - 0.5 * (matrix + matrix.T), 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    tree = gudhi.RipsComplex(distance_matrix=distance, max_edge_length=2.0).create_simplex_tree(max_dimension=2)
    tree.persistence(homology_coeff_field=2, min_persistence=-1.0)
    diagram = np.asarray(tree.persistence_intervals_in_dimension(1), dtype=float)
    if diagram.size == 0:
        return np.empty((0, 2), dtype=float)
    lifetime = diagram[:, 1] - diagram[:, 0]
    return diagram[np.isfinite(lifetime) & (lifetime > 1e-14)]


def landscape(diagram: np.ndarray) -> np.ndarray:
    result = np.zeros((LAYERS, GRID.size), dtype=np.float32)
    if diagram.size == 0:
        return result
    tents = np.maximum(
        np.minimum(GRID[None, :] - diagram[:, 0, None], diagram[:, 1, None] - GRID[None, :]),
        0.0,
    )
    ranked = np.sort(tents, axis=0)[::-1]
    count = min(LAYERS, ranked.shape[0])
    result[:count] = ranked[:count]
    return result


def landscape_speed(values: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    derivative = np.diff(values.astype(float), axis=0) / np.diff(gamma)[:, None, None]
    return np.sqrt(np.trapezoid(np.sum(derivative**2, axis=1), GRID, axis=1) / LAYERS)


def quadratic_vertex(x: np.ndarray, y: np.ndarray, index: int) -> tuple[float, bool]:
    if index == 0 or index == x.size - 1:
        return float(x[index]), True
    coefficients = np.polyfit(x[index - 1:index + 2], y[index - 1:index + 2], 2)
    if coefficients[0] >= 0.0:
        return float(x[index]), True
    vertex = float(-coefficients[1] / (2.0 * coefficients[0]))
    if not x[index - 1] <= vertex <= x[index + 1]:
        return float(x[index]), True
    return vertex, False


def peak(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    maximum = float(np.nanmax(y))
    if not np.isfinite(maximum) or maximum <= 1e-14:
        return {
            "sample": float("nan"), "quadratic": float("nan"),
            "value": maximum, "grid_limited": True, "active": False,
        }
    index = int(np.nanargmax(y))
    vertex, limited = quadratic_vertex(x, y, index)
    return {
        "sample": float(x[index]), "quadratic": vertex,
        "value": float(y[index]), "grid_limited": bool(limited), "active": True,
    }


def normalize(values: np.ndarray) -> np.ndarray:
    maximum = float(np.nanmax(values))
    return values / maximum if maximum > 0.0 else np.zeros_like(values)


def analyze_curve(curve: dict[str, np.ndarray]) -> dict[str, object]:
    gamma = curve["Gamma"]
    mid = 0.5 * (gamma[:-1] + gamma[1:])
    h0_ranked = np.stack([mst_deaths(matrix) for matrix in curve["Cstag"]])
    h0_landscapes = np.stack([
        landscape(np.column_stack((np.zeros(row.size), row))) for row in h0_ranked
    ])
    h1_landscapes = np.stack([landscape(h1_diagram(matrix)) for matrix in curve["Cstag"]])
    ranked_derivative_all = np.abs(np.diff(h0_ranked, axis=0) / np.diff(gamma)[:, None])
    ranked_derivative = ranked_derivative_all[:, :RANKED_K]
    responses = {
        "physical_bulk": -np.diff(curve["bulk_ms2"]) / np.diff(gamma),
        "physical_paper": -np.diff(curve["paper_Cstag"]) / np.diff(gamma),
        "H0_landscape": landscape_speed(h0_landscapes, gamma),
        "H0_ranked_L1_K5": np.mean(ranked_derivative, axis=1),
        "H0_ranked_L2_K5": np.sqrt(np.mean(ranked_derivative**2, axis=1)),
        "H0_ranked_L1_all": np.mean(ranked_derivative_all, axis=1),
        "H0_ranked_L2_all": np.sqrt(np.mean(ranked_derivative_all**2, axis=1)),
        "H1_landscape": landscape_speed(h1_landscapes, gamma),
    }
    for rank_index in range(RANKED_K):
        responses[f"H0_rank_{rank_index + 1}"] = ranked_derivative_all[:, rank_index]
    peaks = {name: peak(mid, values) for name, values in responses.items()}
    return {
        "mid": mid, "responses": responses, "peaks": peaks,
        "H0_ranked": h0_ranked, "H0_landscapes": h0_landscapes,
        "H1_landscapes": h1_landscapes, "H0_rank_velocity": ranked_derivative_all,
    }


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_physics(output: Path, curves: list[dict[str, np.ndarray]], analyses: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(curves)))
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), constrained_layout=True)
    for curve, analysis, color in zip(curves, analyses, colors):
        label = f"h={float(curve['h']):g}"
        axes[0, 0].plot(curve["Gamma"], curve["bulk_ms2"], "o-", ms=2.8, lw=1.1, color=color, label=label)
        axes[0, 1].plot(curve["Gamma"], curve["entropy"], "o-", ms=2.8, lw=1.1, color=color)
        axes[1, 0].plot(analysis["mid"], normalize(analysis["responses"]["physical_bulk"]), "o-", ms=2.8, lw=1.1, color=color)
    h = np.asarray([float(curve["h"]) for curve in curves])
    target = np.asarray([float(curve["target"]) for curve in curves])
    physical = np.asarray([analysis["peaks"]["physical_bulk"]["quadratic"] for analysis in analyses])
    sample = np.asarray([analysis["peaks"]["physical_bulk"]["sample"] for analysis in analyses])
    axes[1, 1].plot(h, target, "o--", color="0.45", label="Fig. 1 digitalizada")
    axes[1, 1].plot(h, physical, "o-", color="black", label=r"$-d m_{s,bulk}^2/d\Gamma$")
    limited = np.asarray([analysis["peaks"]["physical_bulk"]["grid_limited"] for analysis in analyses], dtype=bool)
    axes[1, 1].scatter(h[limited], sample[limited], marker="x", s=60, color="#D55E00", label="máximo limitado")
    axes[1, 1].plot([0, 2], [1, 0], "s", mfc="none", color="0.25", label="extremos exactos")
    axes[0, 0].set(ylabel=r"$m_{s,bulk}^2$", title="Orden antiferromagnético bulk")
    axes[0, 1].set(ylabel=r"$S_{L/2}$", title="Entropía de entrelazamiento")
    axes[1, 0].set(ylabel="respuesta / máximo del corte", title="Respuesta física por corte")
    axes[1, 1].set(xlabel=r"$h$", ylabel=r"$\Gamma_c$", title="Frontera a L=64")
    for axis in axes.flat[:3]:
        axis.set_xlabel(r"$\Gamma$")
    for axis in axes.flat:
        axis.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1, 1].legend(frameon=False, fontsize=8)
    figure.suptitle(r"Modelo limpio OBC, $L=64$, $\chi=128$: datos disponibles combinados")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"physical_L64_overview.{suffix}", dpi=220)
    plt.close(figure)


def plot_responses(output: Path, curves: list[dict[str, np.ndarray]], analyses: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 3, figsize=(13.2, 13.0), constrained_layout=True, sharey=True)
    styles = (
        ("physical_bulk", "física bulk", "black", "-"),
        ("H0_landscape", "H0 landscape", "#0072B2", "-"),
        ("H0_ranked_L1_K5", "H0 barras L1, K=5", "#009E73", "-"),
        ("H0_ranked_L2_K5", "H0 barras L2, K=5", "#56B4E9", "--"),
        ("H0_ranked_L1_all", "H0 todas las barras", "#CC79A7", "-"),
        ("H1_landscape", "H1 landscape", "#D55E00", "-"),
    )
    for axis, curve, analysis in zip(axes.flat, curves, analyses):
        for name, label, color, linestyle in styles:
            axis.plot(analysis["mid"], normalize(analysis["responses"][name]), color=color, linestyle=linestyle, lw=1.45, label=label)
        axis.axvline(float(curve["target"]), color="0.55", linestyle=":", lw=1.0)
        title = f"h={float(curve['h']):g}  ({curve['Gamma'].size} puntos)"
        if np.any(curve["branch_disagreement"]) or float(curve["h"]) == 1.95:
            title += "  [QC]"
        axis.set(title=title, xlabel=r"$\Gamma$ midpoint", ylabel="respuesta normalizada", ylim=(-0.03, 1.08))
        axis.grid(alpha=0.16)
    for axis in axes.flat[len(curves):]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=7, ncol=2)
    figure.suptitle(r"Respuesta física y nuevos observables topológicos, limpio OBC $L=64$")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"topology_responses_by_h.{suffix}", dpi=220)
    plt.close(figure)


def plot_boundary(output: Path, curves: list[dict[str, np.ndarray]], analyses: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    h = np.asarray([float(curve["h"]) for curve in curves])
    target = np.asarray([float(curve["target"]) for curve in curves])
    names = (
        ("physical_bulk", "física bulk", "black", "o"),
        ("H0_landscape", "H0 landscape", "#0072B2", "s"),
        ("H0_ranked_L1_K5", "H0 barras L1, K=5", "#009E73", "^"),
        ("H0_ranked_L1_all", "H0 todas L1", "#CC79A7", "v"),
        ("H0_ranked_L2_all", "H0 todas L2", "#E69F00", "P"),
        ("H1_landscape", "H1 landscape", "#D55E00", "D"),
    )
    physical = np.asarray([a["peaks"]["physical_bulk"]["quadratic"] for a in analyses])
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    axes[0].plot(h, target, "o--", color="0.55", lw=1.2, label="Fig. 1 digitalizada")
    for name, label, color, marker in names:
        values = np.asarray([a["peaks"][name]["quadratic"] for a in analyses])
        axes[0].plot(h, values, marker=marker, color=color, lw=1.25, ms=5, label=label)
        if name != "physical_bulk":
            axes[1].plot(h, values - physical, marker=marker, color=color, lw=1.25, ms=5, label=label)
    axes[0].plot([0, 2], [1, 0], "p", mfc="none", color="0.2", ms=7, label="extremos exactos")
    for axis in axes:
        axis.axvspan(1.925, 1.975, color="#D55E00", alpha=0.09)
        axis.grid(alpha=0.18)
        axis.set_xlabel(r"$h$")
    axes[1].axhline(0.0, color="0.4", lw=0.8)
    axes[0].set(ylabel=r"$\Gamma$ del máximo", title="Frontera inferida por cada detector")
    axes[1].set(ylabel=r"$\Gamma_{topo}-\Gamma_{física}$", title="Desplazamiento respecto al bulk")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2)
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle(r"Comparación de detectores a $L=64$ (zona sombreada: $h=1.95$ no controlado a $\chi=128$)")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"boundary_detector_comparison.{suffix}", dpi=220)
    plt.close(figure)


def plot_ranked_h0(output: Path, curves: list[dict[str, np.ndarray]], analyses: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    colors = plt.cm.Blues(np.linspace(0.42, 0.95, RANKED_K))
    figure, axes = plt.subplots(4, 3, figsize=(13.2, 13.0), constrained_layout=True, sharey=True)
    for axis, curve, analysis in zip(axes.flat, curves, analyses):
        for rank_index, color in enumerate(colors):
            values = analysis["responses"][f"H0_rank_{rank_index + 1}"]
            axis.plot(analysis["mid"], normalize(values), color=color, lw=1.05, label=f"barra {rank_index + 1}")
        axis.plot(analysis["mid"], normalize(analysis["responses"]["H0_ranked_L1_K5"]), color="#009E73", lw=2.0, label="media L1 K=5")
        axis.plot(analysis["mid"], normalize(analysis["responses"]["H0_ranked_L2_K5"]), color="#D55E00", lw=1.45, ls="--", label="RMS L2 K=5")
        axis.axvline(analysis["peaks"]["physical_bulk"]["quadratic"], color="black", ls=":", lw=1.0)
        axis.set(title=f"h={float(curve['h']):g}", xlabel=r"$\Gamma$ midpoint", ylabel="velocidad / máximo", ylim=(-0.03, 1.08))
        axis.grid(alpha=0.16)
    for axis in axes.flat[len(curves):]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=7, ncol=2)
    figure.suptitle(r"Velocidad directa de las cinco barras H0 más largas, limpio OBC $L=64$")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"ranked_H0_responses_by_h.{suffix}", dpi=220)
    plt.close(figure)


def main() -> None:
    args = arguments()
    config_paths = validate_configs()
    curves, sources = load_curves()
    targets = (
        args.output / "physical_L64_overview.png",
        args.output / "topology_responses_by_h.png",
        args.output / "boundary_detector_comparison.png",
        args.output / "ranked_H0_responses_by_h.png",
        args.output / "peak_summary.csv",
        args.output / "analysis_arrays.npz",
        args.output / "provenance.json",
    )
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError("analysis already exists; pass --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)

    analyses = []
    for index, curve in enumerate(curves):
        print(f"topology h={float(curve['h']):g} ({index + 1}/{len(curves)})", flush=True)
        analyses.append(analyze_curve(curve))

    rows = []
    for curve, analysis in zip(curves, analyses):
        row: dict[str, object] = {
            "L": 64, "chi": 128, "bc": "open", "h": float(curve["h"]),
            "point_count": int(curve["Gamma"].size), "Gamma_min": float(curve["Gamma"][0]),
            "Gamma_max": float(curve["Gamma"][-1]), "figure1_Gamma": float(curve["target"]),
            "branch_disagreement_count": int(np.sum(curve["branch_disagreement"])),
            "bond_saturation_count": int(np.sum(curve["bond"] >= 128)),
            "interpretation_flag": "h1.95_chi_not_controlled" if float(curve["h"]) == 1.95 else "",
        }
        for name, result in analysis["peaks"].items():
            for key, value in result.items():
                row[f"{name}_{key}"] = value
        rows.append(row)
    save_csv(args.output / "peak_summary.csv", rows)

    arrays: dict[str, np.ndarray] = {"h_values": np.asarray([float(c["h"]) for c in curves])}
    for index, (curve, analysis) in enumerate(zip(curves, analyses)):
        prefix = f"h{float(curve['h']):.2f}_"
        for key in ("Gamma", "bulk_ms2", "paper_Cstag", "energy_density", "entropy", "bond", "branch_infidelity", "branch_disagreement"):
            arrays[prefix + key] = np.asarray(curve[key])
        arrays[prefix + "Gamma_mid"] = analysis["mid"]
        arrays[prefix + "H0_ranked"] = analysis["H0_ranked"]
        arrays[prefix + "H0_landscapes"] = analysis["H0_landscapes"]
        arrays[prefix + "H1_landscapes"] = analysis["H1_landscapes"]
        arrays[prefix + "H0_rank_velocity"] = analysis["H0_rank_velocity"]
        for name, values in analysis["responses"].items():
            arrays[prefix + name] = values
    np.savez_compressed(args.output / "analysis_arrays.npz", **arrays)

    h_values = np.asarray([float(curve["h"]) for curve in curves])
    controlled = h_values < 1.95
    metrics: dict[str, object] = {}
    detector_names = (
        "physical_bulk", "H0_landscape", "H0_ranked_L1_K5",
        "H0_ranked_L2_K5", "H0_ranked_L1_all", "H0_ranked_L2_all",
        "H1_landscape",
    )
    target_values = np.asarray([float(curve["target"]) for curve in curves])
    physical_values = np.asarray([analysis["peaks"]["physical_bulk"]["quadratic"] for analysis in analyses])
    for name in detector_names:
        values = np.asarray([analysis["peaks"][name]["quadratic"] for analysis in analyses], dtype=float)
        active = np.asarray([analysis["peaks"][name]["active"] for analysis in analyses], dtype=bool)
        limited = np.asarray([analysis["peaks"][name]["grid_limited"] for analysis in analyses], dtype=bool)
        mask = controlled & active & np.isfinite(values)
        target_difference = values[mask] - target_values[mask]
        physical_difference = values[mask] - physical_values[mask]
        interior = mask & ~limited
        metrics[name] = {
            "active_controlled_cut_count": int(np.sum(mask)),
            "interior_controlled_cut_count": int(np.sum(interior)),
            "bias_vs_digitized_Fig1": float(np.mean(target_difference)) if target_difference.size else None,
            "mae_vs_digitized_Fig1": float(np.mean(np.abs(target_difference))) if target_difference.size else None,
            "rmse_vs_digitized_Fig1": float(np.sqrt(np.mean(target_difference**2))) if target_difference.size else None,
            "bias_vs_physical_bulk": float(np.mean(physical_difference)) if physical_difference.size else None,
            "mae_vs_physical_bulk": float(np.mean(np.abs(physical_difference))) if physical_difference.size else None,
        }
    (args.output / "detector_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    all_inputs = config_paths + sources
    provenance = {
        "analysis": "clean L=64 new H0/H1 observables",
        "matrix": "C=Cstag (raw, unconnected)",
        "filtration": "D=1-C, symmetrized and clipped to [0,2]",
        "homology_dimensions": [0, 1],
        "conditions": {"L": 64, "bc": "open", "chi": 128, "disorder": "none"},
        "campaigns": list(CAMPAIGNS),
        "landscape": {"grid": [0.0, 2.0, 401], "layers": LAYERS},
        "ranked_H0": {"K": RANKED_K, "L1": "mean absolute rank velocity", "L2": "RMS rank velocity"},
        "note": "A deterministic clean scan has one trajectory, so mean velocity equals velocity of the mean identically.",
        "qc": "h=1.95 at chi=128 is retained and explicitly flagged, not used as controlled evidence.",
        "inputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in all_inputs],
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    plot_physics(args.output, curves, analyses)
    plot_responses(args.output, curves, analyses)
    plot_boundary(args.output, curves, analyses)
    plot_ranked_h0(args.output, curves, analyses)


if __name__ == "__main__":
    main()
