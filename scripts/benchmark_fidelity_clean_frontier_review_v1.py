"""Referee benchmark (review v1): fidelity susceptibility on the clean h > 0 cuts (saved MPS, no DMRG).

For every cut of benchmark_estimators_clean_frontier_review_v1.py (same selection), the
selected MPS of each Gamma are taken from the source scans (duplicate Gamma: lower
energy, as in the figure merge), and

    chi_F(Gamma_mid) = -2 ln |<psi(Gamma_k)|psi(Gamma_{k+1})>| / (Gamma_{k+1} - Gamma_k)^2

is evaluated between consecutive field values. Its maximum is taken over intervals whose
two end points are allowed by the QC mask of the figure, with quadratic refinement.

Run with the tn_tda environment (TeNPy + h5py):
    python scripts/benchmark_fidelity_clean_frontier_review_v1.py
Output: results/review_v1/benchmark-clean-frontier-v1/fidelity.json
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_estimators_clean_frontier_review_v1 import FIG, GRID_H, OUT, QC_CUTS, REF_G, REF_H, masked_peak  # noqa: E402

WORKERS = 4


def cut_fidelity(args):
    from clean_tfim_tn.io.clean_ising_mps_hdf5 import load_clean_ising_mps_collection as load_mps_collection_hdf5

    L, h, chi, sources = args
    points = {}
    for s in sources:
        p = Path(s) if Path(s).is_absolute() else ROOT / s
        scan = p.parent
        if not (scan / "selected_mps_manifest.json").is_file():
            continue
        man = json.loads((scan / "selected_mps_manifest.json").read_text(encoding="utf-8"))
        with np.load(p) as z:
            energy = np.asarray(z["selected_energy_density"], float)
            branch = np.asarray(z["branch_disagreement"], bool)
            bond = np.asarray(z["selected_max_bond_dimension"], int)
        for i, rec in enumerate(man["point_records"]):
            key = round(float(rec["Gamma0"] if "Gamma0" in rec else rec["Gamma"]), 12)
            cand = {"scan": scan, "mps": man["mps_file"], "state": rec["state_key"], "energy": energy[i],
                    "flag": bool(branch[i] or bond[i] >= chi)}
            if key not in points or cand["energy"] < points[key]["energy"]:
                points[key] = cand
    g = np.array(sorted(points))
    if g.size < 4:
        return (L, h), None
    cache = {}

    def state(k):
        p = points[g[k]]
        f = p["scan"] / p["mps"]
        if f not in cache:
            cache.clear()
            cache[f] = load_mps_collection_hdf5(f)["selected_states"]
        return cache[f][p["state"]]

    # consecutive states may come from different files; keep the previous state in memory
    chiF = np.full(g.size - 1, np.nan)
    prev = state(0)
    for k in range(g.size - 1):
        nxt = state(k + 1)
        ov = abs(complex(prev.overlap(nxt)))
        chiF[k] = -2.0 * np.log(max(ov, 1e-300)) / (g[k + 1] - g[k]) ** 2
        prev = nxt
    flag = np.array([points[x]["flag"] for x in g])
    allowed = ~(flag | np.r_[False, flag[:-1]] | np.r_[flag[1:], False])
    mid = 0.5 * (g[1:] + g[:-1])
    pk = masked_peak(mid, chiF, allowed[1:] & allowed[:-1])
    return (L, h), {"peak": pk, "ref": float(np.interp(h, REF_H, REF_G)), "min_overlap_interval": float(np.exp(-0.5 * np.nanmax(chiF * np.diff(g) ** 2)))}


def main():
    jobs = []
    for compact, sizes, odd in (("clean_frontier_compact.npz", (16, 32, 64), False),
                                ("clean_frontier_oddL_compact.npz", (31, 63), True)):
        z = np.load(FIG / "data_raw" / compact)
        best = {}
        for e in json.loads(str(z["index_json"])):
            key = (e["L"], round(e["h"], 4))
            if e["L"] in sizes and (key not in best or e["chi"] > best[key]["chi"]):
                best[key] = e
        for (L, h), e in sorted(best.items()):
            if (L, h) in QC_CUTS or (not odd and h >= 2 * (1 - 1 / L) - 1e-12) or (odd and round(h, 4) not in GRID_H):
                continue
            jobs.append((L, h, e["chi"], e["sources"]))
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        results = dict(pool.map(cut_fidelity, jobs))
    summary = {}
    for L in sorted({k[0] for k in results}):
        err = [abs(r["peak"]["x"] - r["ref"]) for k, r in results.items() if k[0] == L and r is not None]
        summary[f"L{L}"] = {"MAE": float(np.mean(err)), "max": float(np.max(err)), "n": len(err),
                            "edge": int(sum(r["peak"]["edge"] for k, r in results.items() if k[0] == L and r is not None))}
        print(f"L{L}: chi_F MAE {summary[f'L{L}']['MAE']:.3f} (n={len(err)}, max {summary[f'L{L}']['max']:.3f})", flush=True)
    summary["cuts"] = {f"L{k[0]}_h{k[1]:g}": r for k, r in results.items()}
    (OUT / "fidelity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
