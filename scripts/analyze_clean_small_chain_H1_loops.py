"""Interpret H1 loops in a small clean OBC staggered-correlation network.

This is an exact-diagonalization diagnostic, not a production TN campaign.
It uses Pauli operators and the project Hamiltonian convention

    H = sum_i Z_i Z_{i+1} - Gamma sum_i X_i - h sum_i Z_i

with uniform couplings, OBC, raw correlations, C=Cstag, and D=1-C.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import gudhi
import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "results"
    / "clean_ising_gamma_campaign"
    / "clean-small-OBC-L6-H1-interpretation-v1"
    / "analysis"
)
L = 6
H_VALUES = np.linspace(0.0, 2.5, 51)
GAMMA_VALUES = np.linspace(0.05, 2.5, 50)
TOLERANCE = 1.0e-12


def basis_spins(length: int) -> np.ndarray:
    states = np.arange(2**length, dtype=np.uint64)
    shifts = np.arange(length - 1, -1, -1, dtype=np.uint64)
    return 1.0 - 2.0 * ((states[:, None] >> shifts) & 1).astype(float)


def transverse_operator(length: int) -> sparse.csr_matrix:
    dimension = 2**length
    rows: list[int] = []
    columns: list[int] = []
    for state in range(dimension):
        for site in range(length):
            rows.append(state)
            columns.append(state ^ (1 << (length - 1 - site)))
    return sparse.csr_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(dimension, dimension)
    )


def ground_state_cstag(
    spins: np.ndarray,
    x_sum: sparse.csr_matrix,
    h: float,
    gamma: float,
    initial: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    diagonal = np.sum(spins[:, :-1] * spins[:, 1:], axis=1) - h * np.sum(spins, axis=1)
    hamiltonian = sparse.diags(diagonal, format="csr") - gamma * x_sum
    energies, vectors = eigsh(
        hamiltonian, k=1, which="SA", v0=initial, tol=1.0e-12, maxiter=10000
    )
    state = vectors[:, 0]
    probability = np.abs(state) ** 2
    czz = spins.T @ (probability[:, None] * spins)
    magnetization = probability @ spins
    sites = np.arange(spins.shape[1])
    parity = (-1.0) ** np.abs(sites[:, None] - sites[None, :])
    return czz * parity, magnetization, state, float(energies[0])


def normalized_connected_cstag(cstag: np.ndarray, magnetization: np.ndarray) -> np.ndarray:
    sites = np.arange(cstag.shape[0])
    parity = (-1.0) ** np.abs(sites[:, None] - sites[None, :])
    czz = cstag * parity
    connected = czz - np.outer(magnetization, magnetization)
    variance = np.maximum(1.0 - magnetization**2, 1.0e-14)
    normalized = connected / np.sqrt(np.outer(variance, variance))
    np.fill_diagonal(normalized, 1.0)
    return np.clip(normalized * parity, -1.0, 1.0)


def persistence(cstag: np.ndarray) -> tuple[np.ndarray, object, np.ndarray]:
    correlation = 0.5 * (cstag + cstag.T)
    distance = np.clip(1.0 - correlation, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    tree = gudhi.RipsComplex(
        distance_matrix=distance, max_edge_length=2.0
    ).create_simplex_tree(max_dimension=2)
    tree.persistence(homology_coeff_field=2, min_persistence=-1.0)
    diagram = np.asarray(tree.persistence_intervals_in_dimension(1), dtype=float)
    if diagram.size == 0:
        diagram = np.empty((0, 2), dtype=float)
    lifetime = diagram[:, 1] - diagram[:, 0]
    diagram = diagram[np.isfinite(lifetime) & (lifetime > 1.0e-14)]
    return diagram, tree, distance


def generator_for_bar(
    tree: object, distance: np.ndarray, birth: float, death: float
) -> tuple[tuple[int, int], tuple[int, int]]:
    generators = tree.flag_persistence_generators()[1][0]
    for row in generators:
        birth_edge = (int(row[0]), int(row[1]))
        death_edge = (int(row[2]), int(row[3]))
        if (
            np.isclose(distance[birth_edge], birth, atol=1.0e-10, rtol=0.0)
            and np.isclose(distance[death_edge], death, atol=1.0e-10, rtol=0.0)
        ):
            return tuple(sorted(birth_edge)), tuple(sorted(death_edge))
    raise RuntimeError("could not match the positive H1 bar to its flag generator")


def representative_cycle(
    distance: np.ndarray,
    birth: float,
    death: float,
    birth_edge: tuple[int, int],
) -> tuple[list[int], list[tuple[int, int]]]:
    epsilon = 0.5 * (birth + death)
    graph = nx.Graph()
    graph.add_nodes_from(range(distance.shape[0]))
    for first in range(distance.shape[0]):
        for second in range(first + 1, distance.shape[0]):
            if distance[first, second] <= epsilon + TOLERANCE:
                graph.add_edge(first, second)
    graph.remove_edge(*birth_edge)
    path = nx.shortest_path(graph, source=birth_edge[0], target=birth_edge[1])
    cycle = path + [path[0]]
    edges = [tuple(sorted(pair)) for pair in zip(cycle[:-1], cycle[1:])]
    return cycle, sorted(set(edges))


def validate_against_existing_l8() -> dict[str, float | int]:
    source = (
        ROOT
        / "results"
        / "clean_ising_gamma_campaign"
        / "clean-h0-Gamma-critical-L8-64-chi64-v1"
        / "scans"
        / "L8_h_0_J1_chi64"
        / "continuation_summary.npz"
    )
    with np.load(source, allow_pickle=False) as archive:
        gamma = np.asarray(archive["Gamma_values"], dtype=float)
        reference = np.asarray(archive["Cstag"], dtype=float)
    spins = basis_spins(8)
    x_sum = transverse_operator(8)
    differences = []
    state = None
    positive_h1 = 0
    for value, target in zip(gamma, reference):
        cstag, _, state, _ = ground_state_cstag(spins, x_sum, 0.0, float(value), state)
        differences.append(float(np.max(np.abs(cstag - target))))
        diagram, _, _ = persistence(cstag)
        positive_h1 += int(diagram.shape[0] > 0)
    return {
        "points": int(gamma.size),
        "maximum_absolute_Cstag_difference_ED_vs_DMRG": float(max(differences)),
        "points_with_positive_H1_ED": int(positive_h1),
        "points_with_positive_H1_DMRG": 0,
    }


def plot(
    maximum_lifetime: np.ndarray,
    connected_maximum_lifetime: np.ndarray,
    representative: dict[str, object],
    cstag: np.ndarray,
    distance: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    birth = float(representative["birth"])
    death = float(representative["death"])
    epsilon = 0.5 * (birth + death)
    cycle_edges = {tuple(edge) for edge in representative["cycle_edges"]}
    death_edge = tuple(representative["death_edge"])

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.5), constrained_layout=True)
    image = axes[0, 0].imshow(
        maximum_lifetime,
        origin="lower",
        aspect="auto",
        extent=[GAMMA_VALUES[0], GAMMA_VALUES[-1], H_VALUES[0], H_VALUES[-1]],
        cmap="magma",
    )
    axes[0, 0].plot(representative["Gamma"], representative["h"], "co", ms=6)
    axes[0, 0].set(xlabel=r"$\Gamma$", ylabel=r"$h$", title="Maximum H1 lifetime")
    figure.colorbar(image, ax=axes[0, 0], label="death - birth")

    matrix = axes[0, 1].imshow(cstag, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[0, 1].set(title="Raw staggered correlator at maximum", xlabel="site j", ylabel="site i")
    axes[0, 1].set_xticks(range(L)); axes[0, 1].set_yticks(range(L))
    figure.colorbar(matrix, ax=axes[0, 1], label=r"$C_{stag}(i,j)$")

    graph = nx.Graph()
    graph.add_nodes_from(range(L))
    for first in range(L):
        for second in range(first + 1, L):
            if distance[first, second] <= epsilon + TOLERANCE:
                graph.add_edge(first, second)
    positions = nx.circular_layout(graph)
    other = [edge for edge in graph.edges() if tuple(sorted(edge)) not in cycle_edges]
    nx.draw_networkx_nodes(graph, positions, node_color="#E6E6E6", edgecolors="black", ax=axes[1, 0])
    nx.draw_networkx_labels(graph, positions, ax=axes[1, 0])
    nx.draw_networkx_edges(graph, positions, edgelist=other, edge_color="0.7", width=1.5, ax=axes[1, 0])
    nx.draw_networkx_edges(graph, positions, edgelist=list(cycle_edges), edge_color="#009E73", width=3.0, ax=axes[1, 0])
    axes[1, 0].set_title(f"Correlation graph at epsilon={epsilon:.3f}\n(green: representative H1 cycle)")
    axes[1, 0].axis("off")

    h_index = int(representative["h_index"])
    axes[1, 1].plot(GAMMA_VALUES, maximum_lifetime[h_index], "o-", ms=3, color="#0072B2")
    axes[1, 1].axvline(float(representative["Gamma"]), color="#D55E00", ls="--")
    axes[1, 1].set(
        xlabel=r"$\Gamma$",
        ylabel="maximum H1 lifetime",
        title=f"Clean OBC L={L} at h={float(representative['h']):.2f}",
    )
    axes[1, 1].grid(alpha=0.2)
    text = (
        f"birth={birth:.4f}  (C={1-birth:.4f})\n"
        f"death={death:.4f}  (C={1-death:.4f})\n"
        f"death edge={death_edge}"
    )
    axes[1, 1].text(0.03, 0.96, text, transform=axes[1, 1].transAxes, va="top")
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT / f"clean_small_L{L}_H1_interpretation.{extension}", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)
    for axis, values, title in (
        (axes[0], maximum_lifetime, "Raw Cstag"),
        (axes[1], connected_maximum_lifetime, "Normalized connected Cstag"),
    ):
        panel = axis.imshow(
            values, origin="lower", aspect="auto",
            extent=[GAMMA_VALUES[0], GAMMA_VALUES[-1], H_VALUES[0], H_VALUES[-1]],
            cmap="magma", vmin=0.0,
            vmax=max(float(np.max(maximum_lifetime)), float(np.max(connected_maximum_lifetime))),
        )
        axis.plot(representative["Gamma"], representative["h"], "co", ms=5)
        axis.set(xlabel=r"$\Gamma$", ylabel=r"$h$", title=title)
    figure.colorbar(panel, ax=axes[:2], label="maximum H1 lifetime")

    pairs = [(1, 2), (1, 3), (2, 4), (3, 4), (1, 4), (2, 3)]
    magnetization = np.asarray(representative["magnetization"], dtype=float)
    sites = np.arange(L)
    parity = (-1.0) ** np.abs(sites[:, None] - sites[None, :])
    product = np.outer(magnetization, magnetization) * parity
    connected = cstag - product
    positions = np.arange(len(pairs))
    width = 0.25
    axes[2].bar(positions - width, [cstag[pair] for pair in pairs], width, label="raw")
    axes[2].bar(positions, [product[pair] for pair in pairs], width, label="one-point product")
    axes[2].bar(positions + width, [connected[pair] for pair in pairs], width, label="connected part")
    axes[2].axhline(0.0, color="black", lw=0.7)
    axes[2].set_xticks(positions, [f"{i}-{j}" for i, j in pairs], rotation=45)
    axes[2].set(ylabel="staggered correlation", title="Representative cycle and diagonals")
    axes[2].legend(frameon=False, fontsize=8)
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT / f"clean_small_L{L}_H1_connected_control.{extension}", dpi=220)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    spins = basis_spins(L)
    x_sum = transverse_operator(L)
    maximum_lifetime = np.zeros((H_VALUES.size, GAMMA_VALUES.size), dtype=float)
    bar_count = np.zeros_like(maximum_lifetime, dtype=int)
    connected_maximum_lifetime = np.zeros_like(maximum_lifetime)
    connected_bar_count = np.zeros_like(maximum_lifetime, dtype=int)
    energy = np.zeros_like(maximum_lifetime)
    best: dict[str, object] | None = None
    rows: list[dict[str, object]] = []

    for h_index, h in enumerate(H_VALUES):
        state = None
        for gamma_index, gamma in enumerate(GAMMA_VALUES):
            cstag, magnetization, state, point_energy = ground_state_cstag(
                spins, x_sum, float(h), float(gamma), state
            )
            diagram, tree, distance = persistence(cstag)
            connected_cstag = normalized_connected_cstag(cstag, magnetization)
            connected_diagram, _, _ = persistence(connected_cstag)
            lifetimes = diagram[:, 1] - diagram[:, 0] if diagram.size else np.empty(0)
            maximum = float(np.max(lifetimes)) if lifetimes.size else 0.0
            maximum_lifetime[h_index, gamma_index] = maximum
            bar_count[h_index, gamma_index] = int(diagram.shape[0])
            connected_lifetimes = (
                connected_diagram[:, 1] - connected_diagram[:, 0]
                if connected_diagram.size else np.empty(0)
            )
            connected_maximum_lifetime[h_index, gamma_index] = (
                float(np.max(connected_lifetimes)) if connected_lifetimes.size else 0.0
            )
            connected_bar_count[h_index, gamma_index] = int(connected_diagram.shape[0])
            energy[h_index, gamma_index] = point_energy
            rows.append({
                "h": float(h), "Gamma": float(gamma),
                "positive_H1_bar_count": int(diagram.shape[0]),
                "maximum_H1_lifetime": maximum,
                "total_H1_persistence": float(np.sum(lifetimes)),
                "normalized_connected_positive_H1_bar_count": int(connected_diagram.shape[0]),
                "normalized_connected_maximum_H1_lifetime": (
                    float(np.max(connected_lifetimes)) if connected_lifetimes.size else 0.0
                ),
                "ground_state_energy": point_energy,
            })
            if best is None or maximum > float(best["lifetime"]):
                if not lifetimes.size:
                    continue
                bar_index = int(np.argmax(lifetimes))
                birth, death = map(float, diagram[bar_index])
                birth_edge, death_edge = generator_for_bar(tree, distance, birth, death)
                cycle, cycle_edges = representative_cycle(
                    distance, birth, death, birth_edge
                )
                best = {
                    "h_index": h_index, "Gamma_index": gamma_index,
                    "h": float(h), "Gamma": float(gamma),
                    "birth": birth, "death": death, "lifetime": maximum,
                    "birth_correlation_threshold": 1.0 - birth,
                    "death_correlation_threshold": 1.0 - death,
                    "birth_edge": list(birth_edge), "death_edge": list(death_edge),
                    "cycle_vertices_closed": cycle,
                    "cycle_edges": [list(edge) for edge in cycle_edges],
                    "magnetization": magnetization.tolist(),
                    "normalized_connected_H1_diagram": connected_diagram.tolist(),
                    "Cstag": cstag.copy(), "distance": distance.copy(),
                    "diagram": diagram.copy(),
                }
        print(f"clean ED h {h_index + 1}/{H_VALUES.size}", flush=True)

    if best is None:
        raise RuntimeError("the clean L6 grid contains no positive-persistence H1 bar")
    validation = validate_against_existing_l8()
    cstag = np.asarray(best.pop("Cstag"), dtype=float)
    distance = np.asarray(best.pop("distance"), dtype=float)
    diagram = np.asarray(best.pop("diagram"), dtype=float)

    with (OUTPUT / "clean_small_L6_H1_scan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(
        OUTPUT / "clean_small_L6_H1_scan.npz",
        h_values=H_VALUES, Gamma_values=GAMMA_VALUES,
        maximum_H1_lifetime=maximum_lifetime, positive_H1_bar_count=bar_count,
        normalized_connected_maximum_H1_lifetime=connected_maximum_lifetime,
        normalized_connected_positive_H1_bar_count=connected_bar_count,
        ground_state_energy=energy, representative_Cstag=cstag,
        representative_distance=distance, representative_H1_diagram=diagram,
    )
    report = {
        "schema_version": 1,
        "model": "clean antiferromagnetic Ising chain with uniform longitudinal and transverse fields",
        "Hamiltonian": "sum ZZ - Gamma sum X - h sum Z",
        "boundary_conditions": "OBC",
        "L": L,
        "matrix": "C=Cstag raw",
        "filtration": "D=1-C constructed internally",
        "grid": {"h": H_VALUES.tolist(), "Gamma": GAMMA_VALUES.tolist()},
        "points": int(H_VALUES.size * GAMMA_VALUES.size),
        "points_with_positive_H1": int(np.count_nonzero(bar_count)),
        "points_with_positive_H1_normalized_connected": int(
            np.count_nonzero(connected_bar_count)
        ),
        "points_with_positive_H1_in_both": int(
            np.count_nonzero((bar_count > 0) & (connected_bar_count > 0))
        ),
        "representative_global_maximum": best,
        "existing_L8_h0_validation": validation,
        "interpretive_guardrail": "H1 is a loop in the correlation graph, not a spatial loop or a topological phase invariant.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot(maximum_lifetime, connected_maximum_lifetime, best, cstag, distance)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
