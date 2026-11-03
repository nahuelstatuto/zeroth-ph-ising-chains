"""Referee benchmark (review v1): xi_lambda against simple correlation responses, clean h > 0.

Same cuts, chi selection, exclusions, QC mask and peak rule as Fig. 4
(paper_figures/fig_clean_frontier/make_figure.py): best chi per (L, h); even chains
without h >= 2J(1 - 1/L) and without the QC cut (64, 1.95); maxima only on points that are
unflagged with unflagged neighbours (flag = branch disagreement or saturated bond
dimension); quadratic refinement; derivatives np.gradient(edge_order=2).
Raw matrices are re-read from the sources listed in the compact files and merged with
the rule of the figure (duplicate Gamma: lower-energy state, never averaged).

Estimators (maximum over Gamma):
  phys    -d m^2_{s,B} / dGamma          (paper physical response)
  xi      H0 landscape velocity           (paper)
  xi_nn   same functional on the sorted nearest-neighbour bars 1 - C_{i,i+1}
  dcnn    |d mean_i C_{i,i+1} / dGamma|   (bond-energy response)
  fro     ||dC/dGamma||_F / L             (whole correlation matrix)
  d2e     -d^2 e / dGamma^2               (energy density, selected state)
  ds      |d S_{L/2} / dGamma|            (half-chain entanglement entropy)
MST locality: fraction of H0 merge edges between nearest neighbours that cannot be
replaced by the nearest-neighbour path at the same value (tie-robust non-locality).

Run:  python scripts/benchmark_estimators_clean_frontier_review_v1.py
Output: results/review_v1/benchmark-clean-frontier-v1/summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_clean_frontier_multisize as frontier  # noqa: E402

FIG = ROOT / "paper_figures" / "fig_clean_frontier"
OUT = ROOT / "results" / "review_v1" / "benchmark-clean-frontier-v1"
REF_H = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.9, 1.95, 2.0])
REF_G = np.array([1.0, 0.992, 0.970, 0.931, 0.878, 0.802, 0.712, 0.599, 0.453, 0.278, 0.154, 0.077, 0.0])
QC_CUTS = {(64, 1.95)}
GRID_H = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9}
ESTIMATORS = ("phys", "xi", "xi_nn", "dcnn", "fro", "d2e", "ds")
TRAP = np.trapezoid


def velocity(g, bars, L):
    t = np.linspace(0.0, max(1.0, 1.02 * bars.max()), 2001)
    layers = np.sort(np.maximum(0.0, np.minimum(t[None, None, :], bars[:, :, None] - t[None, None, :])), axis=1)
    d = np.gradient(layers, g, axis=0, edge_order=2)
    return np.sqrt(TRAP((d ** 2).sum(axis=1), t, axis=1) / (L - 1))


def quadratic_peak(x, y):
    A = np.vstack([x ** 2, x, np.ones(3)]).T
    a, b, c = np.linalg.solve(A, y)
    return -b / (2 * a) if a < 0 else x[1]


def masked_peak(g, y, allowed):
    i = int(np.argmax(np.where(allowed & np.isfinite(y), y, -np.inf)))
    edge = i in (0, len(g) - 1) or not (allowed[max(i - 1, 0)] and allowed[min(i + 1, len(g) - 1)])
    return {"x": float(g[i]) if edge else float(quadratic_peak(g[i - 1:i + 2], y[i - 1:i + 2])), "edge": bool(edge)}


def mst_stats(C):
    L = C.shape[0]
    D = np.clip(1.0 - C, 0.0, 2.0) + 1.0
    np.fill_diagonal(D, 0.0)
    T = minimum_spanning_tree(np.triu(D)).tocoo()
    nn_path = np.clip(1.0 - np.diagonal(C, 1), 0.0, 2.0)
    nonlocal_ = 0
    for i, j, w in zip(T.row, T.col, T.data - 1.0):
        i, j = min(i, j), max(i, j)
        if j - i > 1 and nn_path[i:j].max() > w + 1e-12:
            nonlocal_ += 1
    return 1.0 - nonlocal_ / (L - 1)


def load(compact, sizes, odd):
    z = np.load(FIG / "data_raw" / compact)
    best = {}
    for e in json.loads(str(z["index_json"])):
        key = (e["L"], round(e["h"], 4))
        if e["L"] in sizes and (key not in best or e["chi"] > best[key]["chi"]):
            best[key] = e
    cuts = {}
    for (L, h), e in sorted(best.items()):
        if (L, h) in QC_CUTS or (not odd and h >= 2 * (1 - 1 / L) - 1e-12):
            continue
        if odd and round(h, 4) not in GRID_H:
            continue
        scans = []
        for s in e["sources"]:
            p = Path(s) if Path(s).is_absolute() else ROOT / s
            with np.load(p) as a:
                scans.append({"gamma": np.asarray(a["Gamma_values"], float), "cstag": np.asarray(a["Cstag"], float),
                              "energy": np.asarray(a["selected_energy_density"], float),
                              "entropy": np.asarray(a["half_chain_entropy"], float),
                              "bond": np.asarray(a["selected_max_bond_dimension"], int),
                              "branch": np.asarray(a["branch_disagreement"], bool),
                              "energy_diff": np.asarray(a["absolute_energy_difference"], float),
                              "fidelity": np.asarray(a["branch_fidelity"], float)})
        d = frontier.merge(scans)
        g, C = d["gamma"], np.asarray(d["cstag"], float)
        if g.size < 4:
            continue
        flag = np.asarray(d["branch"], bool) | (np.asarray(d["bond"]) >= e["chi"])
        allowed = ~(flag | np.r_[False, flag[:-1]] | np.r_[flag[1:], False])
        a, b = L // 4, 3 * L // 4
        deaths = np.stack([frontier_deaths(m) for m in C])
        nn_bars = np.sort(np.clip(1.0 - np.stack([np.diagonal(m, 1) for m in C]), 0.0, 2.0), axis=1)
        curves = {
            "phys": -np.gradient(C[:, a:b, a:b].mean(axis=(1, 2)), g, edge_order=2),
            "xi": velocity(g, deaths, L),
            "xi_nn": velocity(g, nn_bars, L),
            "dcnn": np.abs(np.gradient(np.stack([np.diagonal(m, 1).mean() for m in C]), g, edge_order=2)),
            "fro": np.linalg.norm(np.gradient(C, g, axis=0, edge_order=2), axis=(1, 2)) / L,
            "d2e": -np.gradient(np.gradient(np.asarray(d["energy"], float), g, edge_order=2), g, edge_order=2),
            "ds": np.abs(np.gradient(np.asarray(d["entropy"], float), g, edge_order=2)),
        }
        ref = float(np.interp(h, REF_H, REF_G))
        locality = np.array([mst_stats(m) for m in C])
        near = np.abs(g - ref) < 0.1
        cuts[(L, h)] = {"ref": ref, "full": bool(e["includes_full_gamma_scan"] or (g.min() <= 0.05 and g.max() >= 1.1)),
                        "peaks": {k: masked_peak(g, v, allowed) for k, v in curves.items()},
                        "mst_local_fraction_near_boundary": float(locality[near].mean()) if near.any() else None,
                        "mst_local_fraction_min": float(locality.min())}
    return cuts


def frontier_deaths(C):
    D = np.clip(1.0 - C, 0.0, 2.0) + 1.0
    np.fill_diagonal(D, 0.0)
    return np.sort(minimum_spanning_tree(np.triu(D)).tocoo().data - 1.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    even = load("clean_frontier_compact.npz", (16, 32, 64), odd=False)
    odd = load("clean_frontier_oddL_compact.npz", (31, 63), odd=True)
    summary = {"estimators": ESTIMATORS, "per_size": {}, "cuts": {}}
    for name, table in (("even", even), ("odd", odd)):
        for L in sorted({k[0] for k in table}):
            rows = {k: c for k, c in table.items() if k[0] == L}
            entry = {}
            for est in ESTIMATORS:
                err = [abs(c["peaks"][est]["x"] - c["ref"]) for c in rows.values()]
                err_full = [abs(c["peaks"][est]["x"] - c["ref"]) for c in rows.values() if c["full"]]
                entry[est] = {"MAE": float(np.mean(err)), "max": float(np.max(err)), "n": len(err),
                              "MAE_full": float(np.mean(err_full)) if err_full else None, "n_full": len(err_full),
                              "edge": int(sum(c["peaks"][est]["edge"] for c in rows.values()))}
            loc = [c["mst_local_fraction_near_boundary"] for c in rows.values() if c["mst_local_fraction_near_boundary"] is not None]
            entry["mst_local_fraction_near_boundary_mean"] = float(np.mean(loc)) if loc else None
            summary["per_size"][f"L{L}"] = entry
            print(f"L{L} ({name}, {len(rows)} cuts): " + "  ".join(f"{e} {entry[e]['MAE']:.3f}" for e in ESTIMATORS)
                  + f" | MST local near boundary {entry['mst_local_fraction_near_boundary_mean']:.3f}", flush=True)
        for (L, h), c in table.items():
            summary["cuts"][f"L{L}_h{h:g}"] = c
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
