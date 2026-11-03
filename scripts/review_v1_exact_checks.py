"""Exact checks quoted in the V1 manuscript (review v1).

1. DMRG against the exact free-fermion solution at h = 0, with the couplings stored in each
   scan: clean L = 16-128 (paper_figures/fig_clean_h0/data_raw), random L = 64
   (50 realizations, coarse campaign) and L = 128 (fine-window campaign). Maximum
   |C_DMRG - C_exact| over unflagged and flagged points (flag = branch disagreement or
   saturated bond dimension).
2. Locality of the H0 merge tree of the clean chain: fraction of minimum-spanning-tree edges
   between nearest neighbours, over every field value and L = 16-512 (exact data of
   results/review_v1/free-fermion-h0-v1/clean_L*.npz).

Run:  python scripts/review_v1_exact_checks.py
Output: results/review_v1/checks-v1/summary.json
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clean_tfim_tn.exact.free_fermion_h0 import cstag_elimination, majorana_state  # noqa: E402

OUT = ROOT / "results" / "review_v1" / "checks-v1"
RANDOM = {64: ("random-h0-Gamma-critical-L64-c128-s0-49-v1", 128),
          128: ("random-h0-L128-fine625-c256-s*", 256)}
CLEAN = {16: 64, 32: 128, 64: 128, 128: 256}


def compare(C_dmrg, J, Gamma):
    W = majorana_state(J, Gamma)[0]
    return float(np.abs(cstag_elimination(W) - C_dmrg).max())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"dmrg_vs_exact": {}, "clean_mst": {}}
    unflag, flag_err = [], []
    for L, chi in CLEAN.items():
        z = np.load(ROOT / "paper_figures" / "fig_clean_h0" / "data_raw" / f"clean_h0_L{L}_chi{chi}.npz")
        bad = z["branch_disagreement"].astype(bool) | (z["selected_max_bond_dimension"] >= chi)
        for k in range(z["Gamma_values"].size):
            e = compare(z["Cstag"][k], np.full(L - 1, float(z["J"])), z["Gamma_by_point"][k])
            (flag_err if bad[k] else unflag).append(e)
    out["dmrg_vs_exact"]["clean"] = {"sizes": list(CLEAN), "points": len(unflag) + len(flag_err),
                                     "max_unflagged": max(unflag), "max_flagged": max(flag_err) if flag_err else None}
    for L, (pattern, chi) in RANDOM.items():
        unflag, flag_err, seeds = [], [], 0
        for f in sorted(glob.glob(str(ROOT / "results" / "random_ising_gamma_ensemble" / pattern / "scans" / "*" / "continuation_summary.npz"))):
            z = np.load(f)
            seeds += 1
            bond = (z["selected_max_bond_dimension"] if "selected_max_bond_dimension" in z.files else
                    np.maximum(z["forward_max_bond_dimension"], z["backward_max_bond_dimension"]))   # appendix rule
            bad = z["branch_disagreement"].astype(bool) | (bond >= chi)
            for k in range(z["Gamma0_values"].size):
                J = z["J"][k] if z["J"].ndim == 2 else z["J"]
                e = compare(z["Cstag"][k], J, z["Gamma"][k])
                (flag_err if bad[k] else unflag).append(e)
        out["dmrg_vs_exact"][f"random_L{L}"] = {"scans": seeds, "points_unflagged": len(unflag), "points_flagged": len(flag_err),
                                                 "max_unflagged": max(unflag), "max_flagged": max(flag_err) if flag_err else None}
        print(L, out["dmrg_vs_exact"][f"random_L{L}"], flush=True)
    fr = []
    for f in sorted((ROOT / "results" / "review_v1" / "free-fermion-h0-v1").glob("clean_L*.npz")):
        z = np.load(f)
        fr.append((int(z["L"]), float((z["sep"] == 1).mean()), float((z["sep"] == 1).all(axis=-1).mean())))
    out["clean_mst"] = {"sizes": [r[0] for r in fr], "nn_edge_fraction_min": min(r[1] for r in fr),
                        "fields_all_nn_fraction_min": min(r[2] for r in fr)}
    print(out, flush=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
