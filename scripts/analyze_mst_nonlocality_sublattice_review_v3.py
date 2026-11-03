"""B7 (review v3, R1-5): is the non-locality of the H0 merge tree at h > 0 a sublattice effect?

Same cuts, merge rule and near-boundary window (|Gamma - ref| < 0.1) as the MST-locality numbers of
Table II and Sec. V (benchmark_estimators_clean_frontier_review_v1: \\BenchNonLocal*). For every
point we take the minimum spanning tree of D = clip(1 - C, 0, 2) and call an edge (i, j), j > i + 1,
non-replaceable when the nearest-neighbour path between i and j has a larger bar,
max_{i<=k<j} (1 - C_{k,k+1}) > D_ij (bm.mst_stats). Each such edge is classified by
  * distance r = j - i, and for even r by the sublattice of its ends: "aligned" when both ends have
    <sigma^z> > 0 (the sublattice polarized along the longitudinal field), "anti" when both are < 0;
  * and the whole analysis is repeated on the connected staggered correlator
    C^c_ij = Cstag_ij - (-1)^|i-j| <Z_i><Z_j>, with <sigma^z_i> from the stored selected MPS
    (test_connected_correlator_even_review_v2.measure, no DMRG).
Validation: the raw non-local fraction reproduces the benchmark summary for every size.
Run (tn_tda):  python scripts/analyze_mst_nonlocality_sublattice_review_v3.py
Output: results/review_v3/mst-nonlocality-sublattice-v1/summary.json
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_estimators_clean_frontier_review_v1 as bf  # noqa: E402
import test_connected_correlator_even_review_v2 as ce  # noqa: E402

OUT = ROOT / "results" / "review_v3" / "mst-nonlocality-sublattice-v1"
SIZES = {16: False, 32: False, 64: False, 31: True, 63: True}
WINDOW = 0.1


def classify(C, mz):
    """Non-replaceable MST edges of D = clip(1 - C) with their distance and sublattice class."""
    L = C.shape[0]
    D = np.clip(1.0 - C, 0.0, 2.0) + 1.0
    np.fill_diagonal(D, 0.0)
    T = minimum_spanning_tree(np.triu(D)).tocoo()
    nn_path = np.clip(1.0 - np.diagonal(C, 1), 0.0, 2.0)
    edges = []
    for i, j, w in zip(T.row, T.col, T.data - 1.0):
        i, j = min(i, j), max(i, j)
        if j - i > 1 and nn_path[i:j].max() > w + 1e-12:
            r = j - i
            if r % 2:
                cls = "odd"
            elif mz[i] > 0 and mz[j] > 0:
                cls = "aligned"
            elif mz[i] < 0 and mz[j] < 0:
                cls = "anti"
            else:
                cls = "mixed"
            edges.append((r, cls))
    return L - 1, edges


def cuts_of(L, odd):
    z = np.load(bf.FIG / "data_raw" / ("clean_frontier_oddL_compact.npz" if odd else "clean_frontier_compact.npz"))
    best = {}
    for e in json.loads(str(z["index_json"])):
        key = (e["L"], round(e["h"], 4))
        if e["L"] == L and (key not in best or e["chi"] > best[key]["chi"]):
            best[key] = e
    out = {}
    for k, e in sorted(best.items()):
        if k in bf.QC_CUTS or (not odd and k[1] >= 2 * (1 - 1 / L) - 1e-12) or (odd and k[1] not in bf.GRID_H):
            continue
        out[k] = e
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allcuts = {L: cuts_of(L, odd) for L, odd in SIZES.items()}
    sources = sorted({s for cuts in allcuts.values() for e in cuts.values() for s in e["sources"]})
    with ProcessPoolExecutor(max_workers=6) as pool:
        measured = {src: scan for src, scan, _ in pool.map(ce.measure, sources)}
    bench = json.loads((ROOT / "results/review_v1/benchmark-clean-frontier-v1/summary.json").read_text())["per_size"]
    summary = {"window": WINDOW, "per_size": {}, "cuts": {}}
    for L, cuts in allcuts.items():
        sign = (-1.0) ** np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
        tot = {kind: {"bonds": 0, "points": 0, "nonlocal": 0, "r2": 0, "r2_aligned": 0, "r2_anti": 0,
                      "even_aligned": 0, "even_anti": 0, "even_mixed": 0, "odd": 0, "r": {}} for kind in ("raw", "conn")}
        loc_raw = []
        cmp_nn = []
        for (LL, h), e in cuts.items():
            g, d = ce.merge([measured[s] for s in e["sources"]])
            if g.size < 4:
                continue
            ref = float(np.interp(h, bf.REF_H, bf.REF_G))
            near = np.flatnonzero(np.abs(g - ref) < WINDOW)
            if not near.size:
                continue
            per_cut = {}
            for kind in ("raw", "conn"):
                n_nl = 0
                for k in near:
                    C = d["cstag"][k] if kind == "raw" else d["cstag"][k] - sign * np.outer(d["mz"][k], d["mz"][k])
                    nb, edges = classify(C, d["mz"][k])
                    t = tot[kind]
                    t["bonds"] += nb
                    t["points"] += 1
                    t["nonlocal"] += len(edges)
                    n_nl += len(edges)
                    for r, cls in edges:
                        t["r"][r] = t["r"].get(r, 0) + 1
                        t["r2"] += r == 2
                        t["r2_aligned"] += r == 2 and cls == "aligned"
                        t["r2_anti"] += r == 2 and cls == "anti"
                        if cls == "odd":
                            t["odd"] += 1
                        else:
                            t[f"even_{cls}"] += 1
                    if kind == "raw":
                        loc_raw.append(1.0 - len(edges) / nb)
                per_cut[kind] = n_nl / (near.size * (L - 1))
            # Sublattice test on the raw correlator: distance-2 pairs on the aligned sublattice that are more
            # correlated than both nearest-neighbour bonds between them.
            above = []
            for k in near:
                C, m = d["cstag"][k], d["mz"][k]
                i = np.arange(L - 2)
                sel = (m[i] > 0) & (m[i + 2] > 0)
                above.append(float(np.mean(C[i, i + 2][sel] > np.minimum(C[i, i + 1], C[i + 1, i + 2])[sel])) if sel.any() else np.nan)
            cmp_nn.append(np.nanmean(above))
            summary["cuts"][f"L{L}_h{h:g}"] = {"ref": ref, "n_near": int(near.size), "nonlocal_fraction": per_cut,
                                               "aligned_r2_above_nn_fraction": float(np.nanmean(above))}
        entry = {}
        for kind, t in tot.items():
            nl = max(t["nonlocal"], 1)
            entry[kind] = {"nonlocal_fraction_of_bonds": t["nonlocal"] / t["bonds"], "points": t["points"],
                           "share_r2": t["r2"] / nl, "share_r2_aligned": t["r2_aligned"] / nl, "share_r2_anti": t["r2_anti"] / nl,
                           "share_even_aligned": t["even_aligned"] / nl, "share_even_anti": t["even_anti"] / nl,
                           "share_even_mixed": t["even_mixed"] / nl, "share_odd": t["odd"] / nl,
                           "distance_histogram": {str(r): n for r, n in sorted(t["r"].items())}}
        # Validation against Table II / Sec. V: the benchmark averages the local fraction over cuts.
        bench_loc = bench[f"L{L}"]["mst_local_fraction_near_boundary_mean"]
        per_cut_loc = [1.0 - c["nonlocal_fraction"]["raw"] for kk, c in summary["cuts"].items() if kk.startswith(f"L{L}_")]
        entry["benchmark_local_fraction"] = bench_loc
        entry["reproduced_local_fraction"] = float(np.mean(per_cut_loc))
        entry["aligned_r2_above_nn_fraction_mean"] = float(np.nanmean(cmp_nn))
        if abs(entry["reproduced_local_fraction"] - bench_loc) > 1e-9:
            print(f"WARNING L{L}: local fraction {entry['reproduced_local_fraction']} differs from benchmark {bench_loc}", flush=True)
        summary["per_size"][f"L{L}"] = entry
        print(f"L{L}: " + json.dumps({k: {q: (round(v, 3) if isinstance(v, float) else v) for q, v in entry[k].items() if q != 'distance_histogram'}
                                      for k in ("raw", "conn")}), entry["reproduced_local_fraction"], bench_loc, flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
