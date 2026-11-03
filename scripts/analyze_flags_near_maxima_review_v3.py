"""B6 (review v3, R2-MC6): where are the DMRG flags of the clean h > 0 scans, and does excluding them move
the maxima of Fig. 6?

Cuts, flags, estimators and peak rule exactly as paper_figures/fig_clean_frontier/make_figure.py:
highest chi per (L, h); even chains without h >= 2J(1 - 1/L) and without the QC cut (64, 1.95);
flag = branch disagreement or bond saturation; a point is a maximum candidate only if it and its two
neighbours are unflagged. For every cut and estimator (physical -d m^2_{s,B}/dGamma and H0 xi_lambda)
we record
  * the flags by kind and their position relative to the reference boundary (Gamma - Gamma_ref);
  * the fraction of flagged points within +-WINDOW of the masked maximum and of the reference;
  * the maximum found without the mask, its shift and whether it falls on a flagged point;
  * for the flagged points near the maximum, the energy difference and infidelity of the two branches,
    read from the continuation summaries listed in the compact file (lower-energy record per Gamma).
Run:  python scripts/analyze_flags_near_maxima_review_v3.py
Output: results/review_v3/flags-near-maxima-v1/summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper_figures/fig_clean_frontier"
sys.path.insert(0, str(FIG.parent / "_common"))
import paper_style as ps  # noqa: E402

OUT = ROOT / "results/review_v3/flags-near-maxima-v1"
REF_H = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.9, 1.95, 2.0])
REF_G = np.array([1.0, 0.992, 0.970, 0.931, 0.878, 0.802, 0.712, 0.599, 0.453, 0.278, 0.154, 0.077, 0.0])
QC_CUTS = {(64, 1.95)}
GRID_H = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9}
WINDOW = 0.05
TRAP = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def h0_velocity(g, deaths, L):
    t = np.linspace(0.0, max(1.0, 1.02 * deaths.max()), 2001)
    layers = np.sort(np.maximum(0.0, np.minimum(t[None, None, :], deaths[:, :, None] - t[None, None, :])), axis=1)
    d = np.gradient(layers, g, axis=0, edge_order=2)
    return np.sqrt(TRAP((d ** 2).sum(axis=1), t, axis=1) / (L - 1))


def masked_peak(g, y, allowed):
    i = int(np.argmax(np.where(allowed, y, -np.inf)))
    edge = i in (0, len(g) - 1) or not (allowed[max(i - 1, 0)] and allowed[min(i + 1, len(g) - 1)])
    x = float(g[i]) if edge else ps.quadratic_peak(g[i - 1:i + 2], y[i - 1:i + 2])["x"]
    return {"x": x, "i": i, "edge_limited": bool(edge)}


def branch_records(sources):
    """Per Gamma, the lower-energy record of the sources: energy difference and infidelity of the branches."""
    rec = {}
    for s in sources:
        p = Path(s) if Path(s).is_absolute() else ROOT / s
        with np.load(p) as a:
            for gam, e, de, f in zip(a["Gamma_values"], a["selected_energy_density"],
                                     a["absolute_energy_difference"], a["branch_fidelity"]):
                k = round(float(gam), 10)
                if k not in rec or e < rec[k][0]:
                    rec[k] = (float(e), float(de), float(f))
    return rec


def analyse(filename, sizes, odd):
    z = np.load(FIG / "data_raw" / filename)
    best = {}
    for e in json.loads(str(z["index_json"])):
        key = (e["L"], round(e["h"], 4))
        if e["L"] in sizes and (key not in best or e["chi"] > best[key]["chi"]):
            best[key] = e
    rows = []
    for (L, h), e in sorted(best.items()):
        p = e["prefix"]
        g = np.asarray(z[f"{p}_Gamma"], float)
        if g.size < 4 or (L, h) in QC_CUTS or (not odd and h >= 2 * (1 - 1 / L) - 1e-12) or (odd and h not in GRID_H):
            continue
        branch = np.asarray(z[f"{p}_branch_disagreement"], bool)
        sat = np.asarray(z[f"{p}_max_bond"]) >= e["chi"]
        flag = branch | sat
        allowed = ~(flag | np.r_[False, flag[:-1]] | np.r_[flag[1:], False])
        ref = float(np.interp(h, REF_H, REF_G))
        curves = {"physical": -np.gradient(np.asarray(z[f"{p}_bulk_ms2"], float), g, edge_order=2),
                  "H0": h0_velocity(g, np.asarray(z[f"{p}_H0_deaths"], float), L)}
        rec = branch_records(e["sources"]) if flag.any() else {}
        row = {"L": L, "h": h, "chi": e["chi"], "points": int(g.size), "reference": ref,
               "branch": int(branch.sum()), "saturated": int(sat.sum()), "flagged": int(flag.sum()),
               "excluded_candidates": int((~allowed).sum()),
               "flag_offsets": (g[flag] - ref).tolist(),
               "near_reference": {"points": int((np.abs(g - ref) <= WINDOW).sum()),
                                  "flagged": int((flag & (np.abs(g - ref) <= WINDOW)).sum())}}
        for kind, y in curves.items():
            m = masked_peak(g, y, allowed)
            u = masked_peak(g, y, np.ones_like(allowed))
            near = np.abs(g - m["x"]) <= WINDOW
            near_flag = np.flatnonzero(near & flag)
            row[kind] = {"masked": m["x"], "unmasked": u["x"], "shift": u["x"] - m["x"],
                         "unmasked_on_excluded": bool(not allowed[u["i"]]),
                         "masked_err": abs(m["x"] - ref), "unmasked_err": abs(u["x"] - ref),
                         "near_points": int(near.sum()), "near_flagged": int(near_flag.size),
                         "near_flagged_energy_diff_max": max((rec[round(float(g[i]), 10)][1] for i in near_flag), default=None),
                         "near_flagged_infidelity_max": max((1 - rec[round(float(g[i]), 10)][2] for i in near_flag), default=None)}
        rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = analyse("clean_frontier_compact.npz", (16, 32, 64), odd=False) + \
        analyse("clean_frontier_oddL_compact.npz", (31, 63), odd=True)
    figsum = json.loads((FIG / "output/fig_clean_frontier_summary.json").read_text(encoding="utf-8"))
    per_size = {}
    for L in (16, 32, 64, 31, 63):
        r = [x for x in rows if x["L"] == L]
        off = np.concatenate([np.asarray(x["flag_offsets"]) for x in r]) if r else np.array([])
        entry = {"cuts": len(r), "points": sum(x["points"] for x in r), "branch": sum(x["branch"] for x in r),
                 "saturated": sum(x["saturated"] for x in r), "flagged": sum(x["flagged"] for x in r),
                 "cuts_with_flags": sum(x["flagged"] > 0 for x in r),
                 "flags_ordered_side": int((off < -WINDOW).sum()), "flags_disordered_side": int((off > WINDOW).sum()),
                 "flags_within_window_of_reference": int((np.abs(off) <= WINDOW).sum()),
                 "flag_offset_median": float(np.median(off)) if off.size else None}
        for kind in ("physical", "H0"):
            np_ = sum(x[kind]["near_points"] for x in r)
            nf = sum(x[kind]["near_flagged"] for x in r)
            entry[kind] = {"near_points": np_, "near_flagged": nf, "near_flagged_pct": 100 * nf / np_ if np_ else None,
                           "cuts_with_near_flags": sum(x[kind]["near_flagged"] > 0 for x in r),
                           "cuts_shifted": sum(abs(x[kind]["shift"]) > 1e-9 for x in r),
                           "max_abs_shift": max((abs(x[kind]["shift"]) for x in r), default=0.0),
                           "MAE_masked": float(np.mean([x[kind]["masked_err"] for x in r])),
                           "MAE_unmasked": float(np.mean([x[kind]["unmasked_err"] for x in r]))}
        per_size[f"L{L}"] = entry
        print(f"L{L}: " + json.dumps(entry), flush=True)
    # Validation: the masked H0 maxima are those of the figure.
    fig_x = {(c["L"], c["h"]): c["H0"]["x"] for c in figsum["cuts"] + figsum["odd_cuts"] if c.get("H0")}
    dev = [abs(x["H0"]["masked"] - fig_x[(x["L"], x["h"])]) for x in rows if (x["L"], x["h"]) in fig_x]
    out = {"window": WINDOW, "per_size": per_size, "cuts": rows,
           "validation": {"matched_cuts": len(dev), "max_abs_dev_from_figure": max(dev) if dev else None}}
    print("validation", out["validation"])
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
