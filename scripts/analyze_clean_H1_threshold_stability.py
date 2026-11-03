"""Fine H1-count thresholds and unsupervised stability diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from analyze_clean_frontier_multisize import PAPER, save_figure


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "results" / "clean_ising_gamma_campaign"
    / "clean-H1-collective-L6-L64-v1" / "analysis" / "H1_collective_long.csv"
)
OUTPUT = (
    ROOT / "results" / "clean_ising_gamma_campaign"
    / "clean-H1-threshold-fine-v1" / "analysis"
)
THRESHOLDS = tuple(np.round(np.arange(0.005, 0.0101, 0.001), 3))
H_VALUES = (1.4, 1.6, 1.7, 1.8, 1.9, 1.95)


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load() -> dict[float, list[dict[str, str]]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["L"]) == 64]
    result: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        h = float(row["h"])
        if h in H_VALUES:
            result.setdefault(h, []).append(row)
    for values in result.values():
        values.sort(key=lambda row: float(row["Gamma"]))
    return result


def peak(values: list[dict[str, str]], threshold: float) -> dict[str, float | bool]:
    gamma = np.asarray([float(row["Gamma"]) for row in values])
    counts = np.asarray([float(row[f"cycle_count_gt_{threshold:g}"]) for row in values])
    difference = np.diff(counts)
    spacing = np.diff(gamma)
    qc = np.asarray([
        as_bool(row["branch_disagreement"]) or as_bool(row["chi_saturated"])
        for row in values
    ])
    valid = ~(qc[:-1] | qc[1:])
    positive = valid & (difference > 0)
    if not np.any(positive):
        return {"location": float("nan"), "sharpness": 0.0, "edge": True, "max_count": float(np.max(counts))}
    derivative = np.where(positive, difference / spacing, -np.inf)
    index = int(np.argmax(derivative))
    total_positive = float(np.sum(difference[positive]))
    return {
        "location": float(0.5 * (gamma[index] + gamma[index + 1])),
        "sharpness": float(difference[index] / total_positive) if total_positive else 0.0,
        "edge": index == 0 or index == gamma.size - 2,
        "max_count": float(np.max(counts)),
    }


def interpolation_residual(h: np.ndarray, location: np.ndarray) -> float:
    residuals = []
    for index in range(1, h.size - 1):
        expected = np.interp(h[index], [h[index - 1], h[index + 1]], [location[index - 1], location[index + 1]])
        residuals.append(abs(location[index] - expected))
    return float(np.median(residuals))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    grouped = load()
    h = np.asarray(H_VALUES)
    locations = np.empty((len(THRESHOLDS), len(H_VALUES)))
    details: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for ti, threshold in enumerate(THRESHOLDS):
        results = [peak(grouped[value], threshold) for value in H_VALUES]
        locations[ti] = [float(result["location"]) for result in results]
        for value, result in zip(H_VALUES, results, strict=True):
            details.append({"threshold": threshold, "h": value, **result})
        diagnostics.append({
            "threshold": threshold,
            "coverage": int(np.sum(np.isfinite(locations[ti]))),
            "edge_peak_count": int(np.sum([bool(result["edge"]) for result in results])),
            "median_jump_fraction": float(np.median([float(result["sharpness"]) for result in results])),
            "median_max_cycle_count": float(np.median([float(result["max_count"]) for result in results])),
            "median_curve_smoothness_residual": interpolation_residual(h, locations[ti]),
        })
    for index, row in enumerate(diagnostics):
        neighbor_changes = []
        if index > 0:
            neighbor_changes.extend(np.abs(locations[index] - locations[index - 1]))
        if index + 1 < len(THRESHOLDS):
            neighbor_changes.extend(np.abs(locations[index] - locations[index + 1]))
        row["median_neighbor_threshold_shift"] = float(np.median(neighbor_changes))
        row["maximum_neighbor_threshold_shift"] = float(np.max(neighbor_changes))

    def write(name: str, rows: list[dict[str, object]]) -> None:
        with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write("H1_fine_threshold_locations.csv", details)
    write("H1_fine_threshold_unsupervised_diagnostics.csv", diagnostics)

    import matplotlib.pyplot as plt
    paper_h = np.asarray(sorted(PAPER), float)
    paper_gamma = np.asarray([PAPER[value] for value in paper_h], float)
    figure, axis = plt.subplots(figsize=(7.5, 5.3), constrained_layout=True)
    axis.plot(paper_h, paper_gamma, "k-", lw=2.4, label="digitized clean frontier (validation only)")
    for threshold, row, color in zip(THRESHOLDS, locations, plt.cm.viridis(np.linspace(0.05, 0.92, len(THRESHOLDS))), strict=True):
        axis.plot(h, row, "o--", color=color, label=fr"$N_1(\ell>{threshold:.3f})$")
    axis.set(xlabel=r"$h$", ylabel=r"$\Gamma$ of maximum count growth", title="Fine H1 lifetime thresholds")
    axis.grid(alpha=0.18); axis.legend(frameon=False, fontsize=8, ncol=2)
    for extension in ("png", "pdf"):
        save_figure(figure, OUTPUT / f"clean_L64_H1_count_growth_thresholds_005_010.{extension}", dpi=260)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), constrained_layout=True)
    image = axes[0].imshow(locations.T, origin="lower", aspect="auto", cmap="viridis",
                           extent=[THRESHOLDS[0] - 0.0005, THRESHOLDS[-1] + 0.0005, 0, len(H_VALUES)])
    axes[0].set_yticks(np.arange(len(H_VALUES)) + 0.5, [f"{value:g}" for value in H_VALUES])
    axes[0].set(xlabel="lifetime threshold", ylabel=r"$h$", title="Selected transition location")
    figure.colorbar(image, ax=axes[0], label=r"$\Gamma_*$")
    axes[1].plot(THRESHOLDS, [row["median_neighbor_threshold_shift"] for row in diagnostics], "o-", label="median shift to neighbour threshold")
    axes[1].plot(THRESHOLDS, [row["maximum_neighbor_threshold_shift"] for row in diagnostics], "^:", label="maximum shift to neighbour threshold")
    axes[1].plot(THRESHOLDS, [row["median_curve_smoothness_residual"] for row in diagnostics], "s--", label="median cross-h smoothness residual")
    axes[1].set(xlabel="lifetime threshold", ylabel=r"variation in $\Gamma_*$", title="Unsupervised stability diagnostics")
    axes[1].grid(alpha=0.18); axes[1].legend(frameon=False, fontsize=8)
    for extension in ("png", "pdf"):
        save_figure(figure, OUTPUT / f"clean_L64_H1_threshold_stability_005_010.{extension}", dpi=260)
    plt.close(figure)

    payload = {
        "status": "exploratory; threshold choice does not use the clean frontier",
        "conditions": {"L": 64, "bc": "open", "matrix": "C=Cstag", "distance": "D=1-C internally"},
        "thresholds": list(THRESHOLDS),
        "diagnostics": diagnostics,
        "interpretation": [
            "No unique optimal threshold exists without defining an external target or a noise model.",
            "A stable threshold plateau is defensible; its midpoint is a convention, not a learned critical scale.",
            "The frontier overlay is shown only after the unsupervised diagnostics and is not used in them.",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
