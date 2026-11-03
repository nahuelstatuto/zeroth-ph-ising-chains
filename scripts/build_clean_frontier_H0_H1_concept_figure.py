"""Build an exploratory clean-frontier figure combining H0 and H1.

The common dataset is exact diagonalization of the clean OBC L=6 chain.
H0 landscape velocities are evaluated both analytically from the H0 deaths
and numerically by quadrature. H1 uses the maximum lifetime already computed
on the same (h, Gamma) grid. This script writes only to results/, never to
paper_figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_clean_frontier_multisize import PAPER, save_figure
from analyze_clean_small_chain_H1_loops import (
    GAMMA_VALUES,
    H_VALUES,
    L,
    basis_spins,
    ground_state_cstag,
    transverse_operator,
)


ROOT = Path(__file__).resolve().parents[1]
H1_SOURCE = (
    ROOT / "results" / "clean_ising_gamma_campaign"
    / "clean-small-OBC-L6-H1-interpretation-v1" / "analysis"
    / "clean_small_L6_H1_scan.npz"
)
H0_L64_SOURCE = (
    ROOT / "results" / "clean_ising_gamma_campaign"
    / "clean-L64-c128-new-topology-v1" / "analysis" / "analysis_arrays.npz"
)
OUTPUT = (
    ROOT / "results" / "clean_ising_gamma_campaign"
    / "clean-frontier-H0-H1-concept-v1" / "analysis"
)
CACHE = OUTPUT / "clean_L6_H0_H1_concept_arrays.npz"
NUMERICAL_GRID = np.linspace(0.0, 2.0, 4001)
H1_THRESHOLDS = (0.001, 0.005, 0.01)
H0_L64_LAYERS = 15


def mst_deaths(cstag: np.ndarray) -> np.ndarray:
    distance = np.clip(1.0 - 0.5 * (cstag + cstag.T), 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    first, second = np.triu_indices(distance.shape[0], 1)
    order = np.argsort(distance[first, second], kind="stable")
    parent = np.arange(distance.shape[0])
    rank = np.zeros(distance.shape[0], dtype=np.int8)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    result: list[float] = []
    for edge in order:
        left, right = find(int(first[edge])), find(int(second[edge]))
        if left == right:
            continue
        if rank[left] < rank[right]:
            left, right = right, left
        parent[right] = left
        if rank[left] == rank[right]:
            rank[left] += 1
        result.append(float(distance[first[edge], second[edge]]))
        if len(result) == distance.shape[0] - 1:
            break
    return np.sort(np.asarray(result))[::-1]


def tent_squared_distance(death_a: float, death_b: float) -> float:
    """Exact integral of the squared difference of two [0, death] tents."""
    small, large = sorted((max(0.0, death_a), max(0.0, death_b)))
    if large < 2.0 * small:
        return 0.5 * small * (large - small) ** 2
    return large**3 / 12.0 - small**3 / 6.0


def exact_velocity(deaths: np.ndarray) -> np.ndarray:
    delta = np.diff(GAMMA_VALUES)
    result = np.empty((deaths.shape[0], deaths.shape[1] - 1), dtype=float)
    for hi in range(deaths.shape[0]):
        for gi in range(deaths.shape[1] - 1):
            squared = sum(
                tent_squared_distance(first, second)
                for first, second in zip(deaths[hi, gi], deaths[hi, gi + 1], strict=True)
            )
            result[hi, gi] = np.sqrt(squared / (L - 1)) / delta[gi]
    return result


def numerical_velocity(deaths: np.ndarray) -> np.ndarray:
    landscapes = np.empty(
        (deaths.shape[0], deaths.shape[1], L - 1, NUMERICAL_GRID.size), dtype=np.float32
    )
    for hi in range(deaths.shape[0]):
        for gi in range(deaths.shape[1]):
            for layer, death in enumerate(deaths[hi, gi]):
                landscapes[hi, gi, layer] = np.maximum(
                    np.minimum(NUMERICAL_GRID, death - NUMERICAL_GRID), 0.0
                )
    derivative = np.diff(landscapes.astype(float), axis=1) / np.diff(GAMMA_VALUES)[None, :, None, None]
    return np.sqrt(
        np.trapezoid(np.sum(derivative**2, axis=2), NUMERICAL_GRID, axis=2) / (L - 1)
    )


def quadratic_peak(x: np.ndarray, values: np.ndarray) -> float:
    index = int(np.nanargmax(values))
    if index == 0 or index == x.size - 1:
        return float(x[index])
    coefficients = np.polyfit(x[index - 1:index + 2], values[index - 1:index + 2], 2)
    if coefficients[0] >= 0:
        return float(x[index])
    vertex = float(-coefficients[1] / (2.0 * coefficients[0]))
    return vertex if x[index - 1] <= vertex <= x[index + 1] else float(x[index])


def calculate() -> dict[str, np.ndarray]:
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}

    spins = basis_spins(L)
    x_sum = transverse_operator(L)
    deaths = np.empty((H_VALUES.size, GAMMA_VALUES.size, L - 1), dtype=float)
    for hi, h_value in enumerate(H_VALUES):
        state = None
        for gi, gamma_value in enumerate(GAMMA_VALUES):
            cstag, _, state, _ = ground_state_cstag(
                spins, x_sum, float(h_value), float(gamma_value), state
            )
            deaths[hi, gi] = mst_deaths(cstag)
        print(f"H0 ED row {hi + 1}/{H_VALUES.size}: h={h_value:g}", flush=True)

    exact = exact_velocity(deaths)
    numerical = numerical_velocity(deaths)
    gamma_mid = 0.5 * (GAMMA_VALUES[:-1] + GAMMA_VALUES[1:])
    exact_peaks = np.asarray([quadratic_peak(gamma_mid, row) for row in exact])
    numerical_peaks = np.asarray([quadratic_peak(gamma_mid, row) for row in numerical])
    arrays = {
        "h_values": H_VALUES,
        "Gamma_values": GAMMA_VALUES,
        "Gamma_midpoints": gamma_mid,
        "H0_deaths": deaths,
        "H0_velocity_closed_form": exact,
        "H0_velocity_numerical": numerical,
        "H0_peak_closed_form": exact_peaks,
        "H0_peak_numerical": numerical_peaks,
        "numerical_landscape_grid": NUMERICAL_GRID,
    }
    np.savez_compressed(CACHE, **arrays)
    return arrays


def h1_onsets(lifetime: np.ndarray) -> dict[float, np.ndarray]:
    result = {}
    for threshold in H1_THRESHOLDS:
        values = np.full(H_VALUES.size, np.nan)
        for hi, row in enumerate(lifetime):
            hits = np.flatnonzero(row > threshold)
            if hits.size:
                values[hi] = GAMMA_VALUES[int(hits[0])]
        result[threshold] = values
    return result


def normalized_rows(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1, keepdims=True)
    return np.divide(values, maximum, out=np.zeros_like(values), where=maximum > 0)


def load_l64_h0() -> list[dict[str, np.ndarray | float]]:
    curves = []
    with np.load(H0_L64_SOURCE, allow_pickle=False) as archive:
        for h_value in np.asarray(archive["h_values"], float):
            prefix = f"h{h_value:.2f}_"
            gamma = np.asarray(archive[prefix + "Gamma"], float)
            mid = np.asarray(archive[prefix + "Gamma_mid"], float)
            deaths = np.asarray(archive[prefix + "H0_ranked"], float)[:, :H0_L64_LAYERS]
            exact = np.empty(gamma.size - 1, float)
            for index in range(gamma.size - 1):
                squared = sum(
                    tent_squared_distance(first, second)
                    for first, second in zip(deaths[index], deaths[index + 1], strict=True)
                )
                exact[index] = np.sqrt(squared / H0_L64_LAYERS) / (gamma[index + 1] - gamma[index])
            numerical = np.asarray(archive[prefix + "H0_landscape"], float)
            curves.append({
                "h": float(h_value), "Gamma": gamma, "mid": mid,
                "closed": exact, "numerical": numerical,
                "closed_peak": quadratic_peak(mid, exact),
                "numerical_peak": quadratic_peak(mid, numerical),
                "qc": bool(np.any(archive[prefix + "branch_disagreement"]))
                or bool(np.any(np.asarray(archive[prefix + "bond"], float) >= 128)),
            })
    return curves


def plot_multisize_concept(lifetime: np.ndarray) -> dict[str, float]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = load_l64_h0()
    onsets = h1_onsets(lifetime)
    paper_h = np.asarray(sorted(PAPER), float)
    paper_gamma = np.asarray([PAPER[value] for value in paper_h], float)
    colors = {0.6: "#0072B2", 1.2: "#009E73", 1.8: "#CC79A7"}

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), constrained_layout=True)
    for target_h, color in colors.items():
        curve = min(curves, key=lambda item: abs(float(item["h"]) - target_h))
        closed = np.asarray(curve["closed"], float)
        numerical = np.asarray(curve["numerical"], float)
        scale = max(float(np.max(closed)), float(np.max(numerical)))
        axes[0].plot(curve["mid"], closed / scale, "-", lw=2, color=color, label=fr"$h={target_h:g}$ closed")
        axes[0].plot(curve["mid"], numerical / scale, "o", ms=4, mfc="white", mec=color, mew=1.1)
        reference = float(np.interp(target_h, paper_h, paper_gamma))
        axes[0].axvline(
            reference, color=color, ls=":", lw=1.2,
            label="clean frontier" if target_h == 0.6 else None,
        )
    axes[0].plot([], [], "o", ms=4, mfc="white", mec="0.25", label="numerical quadrature")
    axes[0].set(
        title=r"(a) $H_0$ landscape velocity ($L=64$)", xlabel=r"transverse field $\Gamma$",
        ylabel="velocity / maximum", xlim=(0.0, 1.08), ylim=(0.0, 1.08),
    )
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2)

    axes[1].plot(paper_h, paper_gamma, "k-", lw=2.5, label="clean frontier")
    regular = [curve for curve in curves if not bool(curve["qc"])]
    flagged = [curve for curve in curves if bool(curve["qc"])]
    axes[1].plot(
        [curve["h"] for curve in regular], [curve["closed_peak"] for curve in regular],
        "s-", ms=5, lw=1.8, color="#0072B2", label=r"$H_0$ peak, closed ($L=64$)",
    )
    axes[1].plot(
        [curve["h"] for curve in regular], [curve["numerical_peak"] for curve in regular],
        "o", ms=4.5, mfc="white", mec="#0072B2", mew=1.1, label=r"$H_0$ numerical",
    )
    if flagged:
        axes[1].plot(
            [curve["h"] for curve in flagged], [curve["closed_peak"] for curve in flagged],
            "s", ms=6, mfc="none", mec="0.45", mew=1.4, label="H0 numerical QC",
        )
    lower, central, upper = onsets[0.001], onsets[0.005], onsets[0.01]
    valid_band = np.isfinite(lower) & np.isfinite(upper) & (H_VALUES <= 1.95)
    valid = np.isfinite(central) & (H_VALUES <= 1.95)
    axes[1].fill_between(H_VALUES[valid_band], lower[valid_band], upper[valid_band], color="#D55E00", alpha=0.18)
    axes[1].plot(H_VALUES[valid], central[valid], "o-", ms=4, lw=1.5, color="#D55E00", label=r"$H_1$ onset ($L=6$)")
    axes[1].annotate(
        r"$H_0$: whole frontier", xy=(0.8, 0.88), xytext=(0.28, 0.52), color="#0072B2",
        arrowprops={"arrowstyle": "->", "color": "#0072B2"},
    )
    axes[1].annotate(
        r"$H_1$: high-$h$ branch", xy=(1.65, 0.42), xytext=(1.02, 0.18), color="#D55E00",
        arrowprops={"arrowstyle": "->", "color": "#D55E00"},
    )
    axes[1].set(
        title="(b) Complementary reconstruction", xlabel=r"longitudinal field $h$",
        ylabel=r"boundary estimate $\Gamma_*$", xlim=(0.15, 2.02), ylim=(0.0, 1.08),
    )
    axes[1].legend(frameon=False, fontsize=7.2, loc="upper right")

    positive = lifetime[lifetime > 0]
    panel = axes[2].pcolormesh(
        H_VALUES, GAMMA_VALUES, lifetime.T, shading="nearest", cmap="Oranges", vmin=0,
        vmax=float(np.quantile(positive, 0.98)),
    )
    axes[2].plot(paper_h, paper_gamma, "k-", lw=2.3, label="clean frontier")
    axes[2].fill_between(H_VALUES[valid_band], lower[valid_band], upper[valid_band], color="#D55E00", alpha=0.2)
    axes[2].plot(H_VALUES[valid], central[valid], "o-", ms=3.2, lw=1.2, color="#D55E00", label=r"$H_1$ onset")
    axes[2].set(
        title=r"(c) $H_1$ activation map ($L=6$)", xlabel=r"longitudinal field $h$",
        ylabel=r"transverse field $\Gamma$", xlim=(0.15, 2.02), ylim=(0.0, 1.08),
    )
    axes[2].legend(frameon=False, fontsize=7.5, loc="upper right")
    figure.colorbar(panel, ax=axes[2], label="maximum H1 lifetime")
    for axis in axes:
        axis.grid(alpha=0.12)
    for extension in ("png", "pdf"):
        save_figure(figure, OUTPUT / f"clean_frontier_H0_H1_concept_v2.{extension}", dpi=260)
    plt.close(figure)

    absolute = np.concatenate([
        np.abs(np.asarray(curve["closed"]) - np.asarray(curve["numerical"])) for curve in curves
    ])
    peak_difference = np.asarray([
        abs(float(curve["closed_peak"]) - float(curve["numerical_peak"])) for curve in curves
    ])
    regular = [curve for curve in curves if not bool(curve["qc"])]
    h0_errors = np.asarray([
        float(curve["closed_peak"]) - float(np.interp(float(curve["h"]), paper_h, paper_gamma))
        for curve in regular
    ])
    h1_window = valid & (H_VALUES >= 1.2)
    h1_reference = np.interp(H_VALUES[h1_window], paper_h, paper_gamma)
    return {
        "L64_maximum_absolute_velocity_difference": float(np.max(absolute)),
        "L64_maximum_peak_position_difference": float(np.max(peak_difference)),
        "L64_H0_closed_form_MAE_vs_digitized_frontier_excluding_QC": float(np.mean(np.abs(h0_errors))),
        "L6_H1_onset_0p005_MAE_vs_digitized_frontier_h1p2_to_1p95": float(
            np.mean(np.abs(central[h1_window] - h1_reference))
        ),
    }


def plot(arrays: dict[str, np.ndarray], lifetime: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = arrays["h_values"]
    gamma_mid = arrays["Gamma_midpoints"]
    exact = arrays["H0_velocity_closed_form"]
    exact_peaks = arrays["H0_peak_closed_form"]
    numerical_peaks = arrays["H0_peak_numerical"]
    onsets = h1_onsets(lifetime)
    paper_h = np.asarray(sorted(PAPER), float)
    paper_gamma = np.asarray([PAPER[value] for value in paper_h], float)

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), constrained_layout=True)

    panel = axes[0].pcolormesh(
        h, gamma_mid, normalized_rows(exact).T, shading="nearest", cmap="Blues", vmin=0, vmax=1
    )
    axes[0].plot(paper_h, paper_gamma, "k-", lw=2.2, label="clean frontier")
    axes[0].plot(h, exact_peaks, color="#0072B2", lw=1.8, label="H0 closed form")
    axes[0].plot(h[::2], numerical_peaks[::2], "o", ms=3.8, mfc="white", mec="#D55E00", mew=1.0, label="H0 numerical")
    axes[0].set(title=r"(a) $H_0$ reconstructs the frontier", xlabel=r"longitudinal field $h$", ylabel=r"transverse field $\Gamma$")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    figure.colorbar(panel, ax=axes[0], label="row-normalized H0 velocity")

    vmax = float(np.quantile(lifetime[lifetime > 0], 0.98))
    panel = axes[1].pcolormesh(
        h, GAMMA_VALUES, lifetime.T, shading="nearest", cmap="Oranges", vmin=0, vmax=vmax
    )
    axes[1].plot(paper_h, paper_gamma, "k-", lw=2.2, label="clean frontier")
    lower, central, upper = onsets[0.001], onsets[0.005], onsets[0.01]
    valid_band = np.isfinite(lower) & np.isfinite(upper)
    axes[1].fill_between(h[valid_band], lower[valid_band], upper[valid_band], color="#D55E00", alpha=0.22, linewidth=0)
    valid = np.isfinite(central)
    axes[1].plot(h[valid], central[valid], "o-", ms=3.2, lw=1.2, color="#D55E00", label=r"H1 onset ($>0.005$)")
    axes[1].set(title=r"(b) $H_1$ activates only at high $h$", xlabel=r"longitudinal field $h$", ylabel=r"transverse field $\Gamma$")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    figure.colorbar(panel, ax=axes[1], label="maximum H1 lifetime")

    axes[2].plot(paper_h, paper_gamma, "k-", lw=2.4, label="clean frontier")
    mask = (h >= paper_h.min()) & (h <= paper_h.max())
    axes[2].plot(h[mask], exact_peaks[mask], "s-", ms=4, lw=1.6, color="#0072B2", label="H0 peak")
    axes[2].plot(h[mask][::2], numerical_peaks[mask][::2], "o", ms=4.2, mfc="white", mec="#0072B2", label="H0 numerical check")
    visible_h1 = valid & mask
    axes[2].plot(h[visible_h1], central[visible_h1], "o-", ms=4, lw=1.6, color="#D55E00", label="H1 onset")
    axes[2].fill_between(h[valid_band & mask], lower[valid_band & mask], upper[valid_band & mask], color="#D55E00", alpha=0.18)
    axes[2].annotate("H0: full frontier", xy=(0.72, 0.89), xytext=(0.35, 0.55), color="#0072B2", arrowprops={"arrowstyle": "->", "color": "#0072B2"})
    axes[2].annotate("H1: high-$h$ branch", xy=(1.62, 0.44), xytext=(1.05, 0.18), color="#D55E00", arrowprops={"arrowstyle": "->", "color": "#D55E00"})
    axes[2].set(title="(c) Complementary information", xlabel=r"longitudinal field $h$", ylabel=r"boundary estimate $\Gamma_*$")
    axes[2].legend(frameon=False, fontsize=7.5, loc="upper right")

    for axis in axes:
        axis.set_xlim(0.15, 2.02); axis.set_ylim(0.0, 1.12); axis.grid(alpha=0.12)
    for extension in ("png", "pdf"):
        save_figure(figure, OUTPUT / f"clean_frontier_H0_H1_concept.{extension}", dpi=260)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    arrays = calculate()
    with np.load(H1_SOURCE, allow_pickle=False) as archive:
        lifetime = np.asarray(archive["maximum_H1_lifetime"], float)
    plot(arrays, lifetime)
    l64_validation = plot_multisize_concept(lifetime)

    exact = arrays["H0_velocity_closed_form"]
    numerical = arrays["H0_velocity_numerical"]
    scale = np.maximum(np.abs(exact), 1e-14)
    onsets = h1_onsets(lifetime)
    summary = {
        "status": "exploratory concept figure; not a paper_figures artifact",
        "conditions": {
            "bc": "open", "model": "clean", "matrix": "C=Cstag",
            "H0": "L=64 DMRG targeted frontier cuts",
            "H1": "L=6 exact-diagonalization full map",
        },
        "H0": {
            "closed_form": (
                "For deaths a<=b: integral (lambda_a-lambda_b)^2 = "
                "a(b-a)^2/2 if b<2a, otherwise b^3/12-a^3/6; summed over all L-1 layers."
            ),
            "numerical_grid": [float(NUMERICAL_GRID[0]), float(NUMERICAL_GRID[-1]), int(NUMERICAL_GRID.size)],
            "maximum_absolute_velocity_difference": float(np.max(np.abs(exact - numerical))),
            "maximum_relative_velocity_difference_above_1e-10": float(
                np.max(np.abs(exact - numerical)[exact > 1e-10] / scale[exact > 1e-10])
            ),
            "maximum_peak_position_difference": float(
                np.max(np.abs(arrays["H0_peak_closed_form"] - arrays["H0_peak_numerical"]))
            ),
            **l64_validation,
        },
        "H1": {
            "observable": "maximum H1 lifetime",
            "onset_thresholds": list(H1_THRESHOLDS),
            "first_h_with_onset": {
                str(threshold): float(H_VALUES[np.flatnonzero(np.isfinite(values))[0]])
                for threshold, values in onsets.items() if np.any(np.isfinite(values))
            },
        },
        "guardrails": [
            "The digitized line is a thermodynamic reference; topology is finite OBC L=6.",
            "H0 row normalization is only for displaying ridge locations, not amplitudes across h.",
            "H1 threshold 0.005 is accompanied by the 0.001--0.01 sensitivity band.",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
