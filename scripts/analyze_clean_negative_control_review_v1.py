"""Negative control of review v1: clean odd chains at h > 2J (no transition along Gamma).

Scans: clean-L{31,63}-oddL-fullgamma-h{22,25}-c128-v1 (enqueue_clean_negative_control_review_v1.py).
Same estimators, QC mask and peak rule as benchmark_estimators_clean_frontier_review_v1.py
(= Fig. 4). Reference frontier cuts at h = 1.7 and 1.9 (same L, same grid) are analysed
identically for comparison. Reported per cut: QC flags, position of the maximum, whether it
is interior, peak height and prominence (maximum over the median of the curve), and the
number of H1 intervals with lifetime above tau = 0.04.

Run:  python scripts/analyze_clean_negative_control_review_v1.py
Output: results/review_v1/negative-control-v1/summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_estimators_clean_frontier_review_v1 import frontier_deaths, masked_peak, velocity  # noqa: E402

BASE = ROOT / "results" / "clean_ising_gamma_campaign"
OUT = ROOT / "results" / "review_v1" / "negative-control-v1"
CUTS = {(L, h): BASE / f"clean-L{L}-oddL-fullgamma-h{str(h).replace('.', '')}-c128-v1" / "scans" / f"L{L}_h_{h:g}_J1_chi128"
        for L in (31, 63) for h in (1.7, 1.9, 2.2, 2.5)}
TAU = 0.04


def h1_count(C):
    import gudhi
    D = np.clip(1.0 - 0.5 * (C + C.T), 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    st = gudhi.RipsComplex(distance_matrix=D, max_edge_length=2.0).create_simplex_tree(max_dimension=2)
    st.persistence(homology_coeff_field=2, min_persistence=-1.0)
    iv = np.asarray(st.persistence_intervals_in_dimension(1), float).reshape(-1, 2)
    return int(np.sum((iv[:, 1] - iv[:, 0]) > TAU)) if iv.size else 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for (L, h), d in CUTS.items():
        z = np.load(d / "continuation_summary.npz")
        g, C = np.asarray(z["Gamma_values"], float), np.asarray(z["Cstag"], float)
        flag = np.asarray(z["branch_disagreement"], bool) | (np.asarray(z["selected_max_bond_dimension"]) >= 128)
        allowed = ~(flag | np.r_[False, flag[:-1]] | np.r_[flag[1:], False])
        a, b = L // 4, 3 * L // 4
        curves = {
            "phys": -np.gradient(C[:, a:b, a:b].mean(axis=(1, 2)), g, edge_order=2),
            "xi": velocity(g, np.stack([frontier_deaths(m) for m in C]), L),
            "dcnn": np.abs(np.gradient(np.stack([np.diagonal(m, 1).mean() for m in C]), g, edge_order=2)),
            "ds": np.abs(np.gradient(np.asarray(z["half_chain_entropy"], float), g, edge_order=2)),
        }
        row = {"qc_flags": int(flag.sum()), "points": int(g.size)}
        for k, y in curves.items():
            pk = masked_peak(g, y, allowed)
            row[k] = {"x": pk["x"], "edge": pk["edge"], "height": float(np.max(y[allowed])),
                      "prominence": float(np.max(y[allowed]) / np.median(y[allowed]))}
        n1 = np.array([h1_count(m) for m in C])
        row["H1_max_count"] = int(n1.max())
        row["H1_first_rise"] = float(g[int(np.argmax(np.diff(n1) >= 1)) + 1]) if (np.diff(n1) >= 1).any() else None
        summary[f"L{L}_h{h:g}"] = row
        print(f"L{L} h={h}: flags {row['qc_flags']} | " + "  ".join(
            f"{k} x={row[k]['x']:.3f}{'(edge)' if row[k]['edge'] else ''} H={row[k]['height']:.3f} P={row[k]['prominence']:.1f}"
            for k in curves) + f" | H1 max {row['H1_max_count']} first {row['H1_first_rise']}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
