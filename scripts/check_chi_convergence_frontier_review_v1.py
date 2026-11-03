"""Referee 2 (Sec. 5): convergence in the bond dimension near the maxima at h > 0 (existing data only).

1. For every cut of Fig. 5 (even and odd chains, the bond dimension chi of the paper), the largest
   bond dimension reached by the selected states within |Gamma - reference| < 0.1, relative to chi.
   Below chi the truncation is set by the singular-value threshold of the protocol and not by chi,
   so a larger chi gives the same state.
2. The only usable cut scanned with two bond dimensions, L = 32 at h = 1.9 (chi = 128 and 256):
   maxima of xi_lambda and of the physical response with both.
Run:  python scripts/check_chi_convergence_frontier_review_v1.py
Output: results/review_v1/chi-convergence-v1/summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_estimators_clean_frontier_review_v1 as bf  # noqa: E402

OUT = ROOT / "results" / "review_v1" / "chi-convergence-v1"
FIG = ROOT / "paper_figures" / "fig_clean_frontier" / "data_raw"
WINDOW = 0.1


def cuts(name):
    z = np.load(FIG / name)
    best = {}
    for e in json.loads(str(z["index_json"])):
        key = (e["L"], round(e["h"], 4))
        if key in bf.QC_CUTS:
            continue
        if key not in best or e["chi"] > best[key]["chi"]:
            best[key] = e
    return z, best


def peaks(z, e):
    p, L = e["prefix"], e["L"]
    g = np.asarray(z[f"{p}_Gamma"], float)
    flag = np.asarray(z[f"{p}_branch_disagreement"], bool) | (np.asarray(z[f"{p}_max_bond"]) >= e["chi"])
    allowed = ~(flag | np.r_[False, flag[:-1]] | np.r_[flag[1:], False])
    xi = bf.velocity(g, np.asarray(z[f"{p}_H0_deaths"], float), L)
    phys = -np.gradient(np.asarray(z[f"{p}_bulk_ms2"], float), g, edge_order=2)
    return {"xi": bf.masked_peak(g, xi, allowed)["x"], "phys": bf.masked_peak(g, phys, allowed)["x"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("clean_frontier_compact.npz", "clean_frontier_oddL_compact.npz"):
        z, best = cuts(name)
        for (L, h), e in sorted(best.items()):
            if h > 1.98 or (L % 2 == 0 and h >= 2.0 * (1.0 - 1.0 / L)):   # cuts excluded in Sec. V
                continue
            p = e["prefix"]
            g = np.asarray(z[f"{p}_Gamma"], float)
            ref = float(np.interp(h, bf.REF_H, bf.REF_G))
            near = np.abs(g - ref) < WINDOW
            if not near.any():
                continue
            mb = np.asarray(z[f"{p}_max_bond"])[near]
            rows.append({"L": L, "h": h, "chi": e["chi"], "max_bond_near": int(mb.max()),
                         "ratio": float(mb.max() / e["chi"]), "saturated_near": int((mb >= e["chi"]).sum()),
                         "points_near": int(near.sum())})
    z, _ = cuts("clean_frontier_compact.npz")
    pair = {e["chi"]: e for e in json.loads(str(z["index_json"])) if e["L"] == 32 and round(e["h"], 4) == 1.9}
    two = {chi: peaks(z, e) for chi, e in sorted(pair.items())}
    out = {"window": WINDOW, "cuts": rows,
           "max_ratio": max(r["ratio"] for r in rows),
           "cuts_with_saturation_near": sum(r["saturated_near"] > 0 for r in rows), "n_cuts": len(rows),
           "L32_h1.9": two,
           "L32_h1.9_shift": {k: abs(two[256][k] - two[128][k]) for k in ("xi", "phys")}}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "cuts"}, indent=1))
    for r in rows:
        if r["ratio"] > 0.8:
            print(r)


if __name__ == "__main__":
    main()
