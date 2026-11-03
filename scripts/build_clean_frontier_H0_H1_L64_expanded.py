"""Rebuild the exploratory clean OBC L64 H0/H1 frontier with expanded scans."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_clean_frontier_multisize import PAPER, save_figure
from analyze_clean_L64_new_topology import (
    LAYERS,
    h1_diagram,
    landscape,
    landscape_speed,
    mst_deaths,
)
from build_clean_frontier_H0_H1_concept_figure import quadratic_peak, tent_squared_distance


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "clean_ising_gamma_campaign"
OUTPUT = BASE / "clean-frontier-H0-H1-L64-expanded-v1" / "analysis"
CAMPAIGNS = (
    "clean-fig1-bound-L64-c128-v1",
    "clean-fig1-refine-L64-c128-v1",
    "clean-fig1-refine2-L64-c128-v1",
    "clean-L64-H1-active-grid-extension-c128-v1",
    "clean-L64-fullgamma-newh-fine010-c128-v1",
)
FULL_GAMMA_H = (0.3, 0.7, 1.1, 1.3, 1.7)
THRESHOLDS = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10)
CONFIG_KEYS = (
    "model", "disorder", "J", "bc", "conserve", "chi_max", "initial_states",
    "two_site_sweeps", "one_site_sweeps", "max_E_err", "max_S_err", "svd_min",
    "mixer", "branch_energy_tolerance", "branch_infidelity_tolerance",
    "selection_energy_tolerance", "mps_save_policy",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_configs() -> list[Path]:
    paths = [BASE / name / "campaign_config.json" for name in CAMPAIGNS]
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference = configs[0]
    for path, config in zip(paths[1:], configs[1:], strict=True):
        differences = [key for key in CONFIG_KEYS if config.get(key) != reference.get(key)]
        if differences:
            raise ValueError(f"incompatible campaign {path.parent.name}: {differences}")
    if reference.get("bc") != "open" or reference.get("chi_max") != 128:
        raise ValueError("expected clean OBC L64 chi128 inputs")
    return paths


def load_curves() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records: dict[float, dict[float, dict[str, object]]] = {}
    overlaps: list[dict[str, object]] = []
    for campaign_name in CAMPAIGNS:
        for path in sorted((BASE / campaign_name / "scans").glob("*/continuation_summary.npz")):
            with np.load(path, allow_pickle=False) as archive:
                h = round(float(archive["h"]), 12)
                gamma = np.asarray(archive["Gamma_values"], float)
                cstag = np.asarray(archive["Cstag"], float)
                saved_l = int(archive["L"]) if "L" in archive.files else int(cstag.shape[1])
                if saved_l != 64 or cstag.shape[1:] != (64, 64):
                    continue
                per_h = records.setdefault(h, {})
                for index, value in enumerate(gamma):
                    key = round(float(value), 12)
                    record = {
                        "Cstag": cstag[index],
                        "energy_density": float(archive["selected_energy_density"][index]),
                        "bond": int(archive["selected_max_bond_dimension"][index]),
                        "branch_disagreement": bool(archive["branch_disagreement"][index]),
                        "source": str(path.resolve()),
                        "campaign": campaign_name,
                    }
                    if key in per_h:
                        previous = per_h[key]
                        overlaps.append({
                            "h": h,
                            "Gamma": key,
                            "old_campaign": previous["campaign"],
                            "new_campaign": campaign_name,
                            "max_abs_Cstag_difference": float(np.max(np.abs(
                                np.asarray(previous["Cstag"], float) - cstag[index]
                            ))),
                            "abs_energy_density_difference": abs(
                                float(previous["energy_density"]) - record["energy_density"]
                            ),
                        })
                    per_h[key] = record  # Newer campaign wins at deliberate overlaps.

    curves: list[dict[str, object]] = []
    for h in sorted(records):
        points = records[h]
        ordered = sorted(points)
        gamma = np.asarray(ordered, float)
        matrices = np.stack([np.asarray(points[value]["Cstag"], float) for value in ordered])
        curves.append({
            "h": h,
            "Gamma": gamma,
            "Cstag": matrices,
            "bond": np.asarray([points[value]["bond"] for value in ordered], int),
            "branch_disagreement": np.asarray(
                [points[value]["branch_disagreement"] for value in ordered], bool
            ),
            "sources": sorted({str(points[value]["source"]) for value in ordered}),
        })
    return curves, overlaps


def analyze_curve(curve: dict[str, object]) -> dict[str, object]:
    gamma = np.asarray(curve["Gamma"], float)
    matrices = np.asarray(curve["Cstag"], float)
    deaths = np.stack([mst_deaths(matrix)[:LAYERS] for matrix in matrices])
    h0_landscapes = np.stack([
        landscape(np.column_stack((np.zeros(row.size), row))) for row in deaths
    ])
    h0_numerical = landscape_speed(h0_landscapes, gamma)
    h0_closed = np.empty(gamma.size - 1, float)
    for index in range(gamma.size - 1):
        squared = sum(
            tent_squared_distance(first, second)
            for first, second in zip(deaths[index], deaths[index + 1], strict=True)
        )
        h0_closed[index] = np.sqrt(squared / LAYERS) / (gamma[index + 1] - gamma[index])
    midpoint = 0.5 * (gamma[:-1] + gamma[1:])
    point_qc = np.asarray(curve["branch_disagreement"], bool) | (
        np.asarray(curve["bond"], int) >= 128
    )
    safe_interval = ~(point_qc[:-1] | point_qc[1:])

    def safe_peak(values: np.ndarray) -> float:
        if not np.any(safe_interval):
            return quadratic_peak(midpoint, values)
        masked = np.where(safe_interval, values, -np.inf)
        index = int(np.argmax(masked))
        if (
            0 < index < midpoint.size - 1
            and safe_interval[index - 1]
            and safe_interval[index + 1]
        ):
            return quadratic_peak(midpoint, np.where(safe_interval, values, np.nan))
        return float(midpoint[index])

    h1_lifetime = np.empty(gamma.size, float)
    for index, matrix in enumerate(matrices):
        diagram = h1_diagram(matrix)
        h1_lifetime[index] = (
            0.0 if diagram.size == 0 else float(np.max(diagram[:, 1] - diagram[:, 0]))
        )

    qc = bool(np.any(point_qc))
    return {
        "midpoint": midpoint,
        "H0_closed": h0_closed,
        "H0_numerical": h0_numerical,
        "H0_closed_peak": safe_peak(h0_closed),
        "H0_numerical_peak": safe_peak(h0_numerical),
        "H0_excluded_intervals": int(np.sum(~safe_interval)),
        "H1_lifetime": h1_lifetime,
        "qc": qc,
    }


def onset(gamma: np.ndarray, lifetime: np.ndarray, threshold: float) -> tuple[float, bool]:
    hits = np.flatnonzero(lifetime > threshold)
    if not hits.size:
        return float("nan"), False
    index = int(hits[0])
    return float(gamma[index]), index == 0


def plot_threshold(
    rows: list[dict[str, object]], threshold: float, paper_h: np.ndarray, paper_gamma: np.ndarray
) -> None:
    import matplotlib.pyplot as plt

    key = f"H1_onset_gt_{threshold:g}"
    censor_key = key + "_left_censored"
    figure, axis = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    axis.plot(paper_h, paper_gamma, "k-", lw=2.6, label="clean frontier")

    regular = [row for row in rows if not row["QC_flag"]]
    flagged = [row for row in rows if row["QC_flag"]]
    axis.plot(
        [row["h"] for row in regular], [row["H0_closed_peak_Gamma"] for row in regular],
        "s-", ms=5.2, lw=1.7, color="#0072B2", label=r"$H_0$ peak (closed form)",
    )
    axis.plot(
        [row["h"] for row in regular], [row["H0_numerical_peak_Gamma"] for row in regular],
        "o", ms=4.8, mfc="white", mec="#0072B2", mew=1.1, label=r"$H_0$ numerical",
    )
    if flagged:
        axis.plot(
            [row["h"] for row in flagged], [row["H0_closed_peak_Gamma"] for row in flagged],
            "s", ms=6.2, mfc="none", mec="0.45", mew=1.4, label="QC flagged",
        )

    h = np.asarray([row["h"] for row in rows], float)
    raw = np.asarray([row[key] for row in rows], float)
    active = np.isfinite(raw)
    censored = np.asarray([row[censor_key] for row in rows], bool) & active
    resolved = active & ~censored
    if np.any(resolved):
        axis.plot(h[resolved], raw[resolved], "o-", ms=5.2, lw=1.6, color="#D55E00", label=r"$H_1$ onset")
    if np.any(censored):
        axis.plot(h[censored], raw[censored], "^", ms=6.2, color="#D55E00", label=r"$H_1$ left-censored")
    if np.any(~active):
        axis.plot(h[~active], np.zeros(np.sum(~active)), "x", ms=6, mew=1.5, color="#D55E00", label=r"no $H_1$ above threshold")

    completed_full = np.asarray([bool(row["full_Gamma_completed"]) for row in rows])
    if np.any(completed_full):
        axis.scatter(
            h[completed_full], np.full(np.sum(completed_full), 1.045), marker="v", s=30,
            facecolors="none", edgecolors="#009E73", label=r"new full-$\Gamma$ cut",
        )
    axis.set(
        xlabel=r"longitudinal field $h$", ylabel=r"boundary estimate / $H_1$ onset $\Gamma_*$",
        title=fr"Clean OBC $L=64$: $H_1$ lifetime threshold $>{threshold:g}$",
        xlim=(0.15, 2.02), ylim=(-0.025, 1.08),
    )
    axis.grid(alpha=0.16)
    axis.legend(frameon=False, fontsize=7.6, loc="upper right")
    slug = f"{threshold:g}".replace(".", "p")
    for extension in ("png", "pdf"):
        save_figure(figure, OUTPUT / f"clean_frontier_H0_H1_L64_expanded_threshold_{slug}.{extension}", dpi=280)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config_paths = validate_configs()
    curves, overlaps = load_curves()
    analyses = []
    for index, curve in enumerate(curves, start=1):
        analysis = analyze_curve(curve)
        analyses.append(analysis)
        print(
            f"[{index}/{len(curves)}] h={float(curve['h']):g}, "
            f"points={len(np.asarray(curve['Gamma']))}, "
            f"max H1={np.max(np.asarray(analysis['H1_lifetime'])):.6g}",
            flush=True,
        )

    paper_h = np.asarray(sorted(PAPER), float)
    paper_gamma = np.asarray([PAPER[value] for value in paper_h], float)
    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for curve, analysis in zip(curves, analyses, strict=True):
        h = float(curve["h"])
        gamma = np.asarray(curve["Gamma"], float)
        lifetime = np.asarray(analysis["H1_lifetime"], float)
        row: dict[str, object] = {
            "h": h,
            "Gamma_min": float(gamma[0]),
            "Gamma_max": float(gamma[-1]),
            "Gamma_points": int(gamma.size),
            "digitized_frontier_Gamma": float(np.interp(h, paper_h, paper_gamma)),
            "H0_closed_peak_Gamma": float(analysis["H0_closed_peak"]),
            "H0_numerical_peak_Gamma": float(analysis["H0_numerical_peak"]),
            "H1_maximum_lifetime": float(np.max(lifetime)),
            "QC_flag": bool(analysis["qc"]),
            "H0_excluded_QC_intervals": int(analysis["H0_excluded_intervals"]),
            "full_Gamma_completed": bool(
                any(abs(h - value) < 1e-12 for value in FULL_GAMMA_H)
                and gamma[0] <= 1e-14 and gamma[-1] >= 1.2 - 1e-14
            ),
        }
        for threshold in THRESHOLDS:
            value, censored = onset(gamma, lifetime, threshold)
            row[f"H1_onset_gt_{threshold:g}"] = value
            row[f"H1_onset_gt_{threshold:g}_left_censored"] = censored
        rows.append(row)
        prefix = f"h{h:.2f}_"
        arrays[prefix + "Gamma"] = gamma
        arrays[prefix + "H1_lifetime"] = lifetime
        arrays[prefix + "Gamma_mid"] = np.asarray(analysis["midpoint"], float)
        arrays[prefix + "H0_closed"] = np.asarray(analysis["H0_closed"], float)
        arrays[prefix + "H0_numerical"] = np.asarray(analysis["H0_numerical"], float)

    np.savez_compressed(OUTPUT / "analysis_arrays.npz", **arrays)
    with (OUTPUT / "clean_frontier_H0_H1_L64_expanded.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for threshold in THRESHOLDS:
        plot_threshold(rows, threshold, paper_h, paper_gamma)

    completed_full = sorted(float(row["h"]) for row in rows if row["full_Gamma_completed"])
    pending_full = sorted(value for value in FULL_GAMMA_H if value not in completed_full)
    summary = {
        "status": "exploratory complete update with all five preregistered full-Gamma cuts; not for paper_figures",
        "conditions": {"model": "clean", "bc": "open", "L": 64, "chi": 128, "matrix": "C=Cstag"},
        "campaigns": list(CAMPAIGNS),
        "completed_full_Gamma_h": completed_full,
        "pending_full_Gamma_h": pending_full,
        "thresholds": list(THRESHOLDS),
        "overlap_QC": overlaps,
        "rows": rows,
        "provenance": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in config_paths
        ] + [
            {"path": source, "sha256": sha256(Path(source))}
            for source in sorted({source for curve in curves for source in curve["sources"]})
        ],
        "guardrails": [
            "Only completed OBC L64 scans are included.",
            "C=Cstag is the stored topological matrix; D=1-C is used only internally.",
            "QC flags are retained and include branch disagreements or chi=128 saturation.",
            "All five preregistered full-Gamma cuts h=0.3,0.7,1.1,1.3,1.7 are included.",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "h_values": [row["h"] for row in rows],
        "completed_full_Gamma_h": completed_full,
        "pending_full_Gamma_h": pending_full,
    }, indent=2))


if __name__ == "__main__":
    main()
