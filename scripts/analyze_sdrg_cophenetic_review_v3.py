"""B3 (review v3, R1-2): tree-to-tree comparison of the H0 dendrogram with the SDRG decimation hierarchy.

R1: "the SDRG produces its own hierarchy: bond decimations merge clusters at logarithmic scales.
Comparing the H0 dendrogram with a flat final partition wastes information; a cophenetic correlation
or a tree distance would be more natural."

SDRG hierarchy. In the flow of compare_sdrg_h0_clusters_review_v1.sdrg_partition every bond
decimation joins two parts of a future cluster; run_sdrg_rg_scale_purity_review_v3.sdrg_flow stores
for every pair (i, j) the stage rho = (active clusters)/L at which it was joined (nan if i and j end
in different clusters). Since the RG scale Omega decreases monotonically along the flow, rho orders
the joins exactly as ln Omega. The SDRG cophenetic height is h_SDRG(i, j) = -rho_join, and pairs of
different final clusters, which the SDRG never joins, sit above every join (a common root).

Trees compared (heights = cophenetic distances):
  H0        single linkage of D = clip(1 - C, 0, 2): minimax distance along the MST
  average   average linkage of D
  NN        single linkage restricted to nearest-neighbour links: max_{i<=k<j} D_{k,k+1}
  distance  heights |i - j| (no correlations): the SDRG joins far sites late, so this is the
            trivial reference for the order of the decimations
  baseline  the H0 tree against the SDRG hierarchy of an independent realization
Scores per realization (rank based, so no choice of scale is needed for the SDRG heights):
  auc     cluster level: probability that a pair of the same final SDRG cluster is lower in the tree
          than a pair of different clusters (Mann-Whitney, ties 1/2)
  within  internal order: Spearman between tree heights and SDRG join order inside every cluster with
          >= 3 sites, weighted by its number of pairs
  intra   Spearman over all pairs joined by the SDRG, pooled over clusters
  all     Spearman over all pairs, never-joined pairs tied at the top (cophenetic correlation in rank
          form; dominated by the cluster level)
Ensembles and seeds as run_sdrg_rg_scale_purity_review_v3: box (fixed_mean, Sec. IV B) L = 32-512,
seeds[0:300], circular baseline; fixed_logmean power-law family of C1, delta = 0.25-2, L = 32-256,
seeds[1:301]. Validation: the H0 dendrogram purity of the box ensemble reproduces the per-realization
values of the paper figure (paper_figures/fig_sdrg_purity/data_raw/sdrg_hierarchy_box.npz).
OBC, h = 0, Gamma0 = 1, exact free fermions, raw C = Cstag, D = 1 - C.
Run (tn_tda):  python scripts/analyze_sdrg_cophenetic_review_v3.py --workers 12
Output: results/review_v3/sdrg-cophenetic-v1/{tag}.npz + summary.json (blocks skipped on restart).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from clean_tfim_tn.exact.free_fermion_h0 import cstag_elimination, majorana_state  # noqa: E402
from compare_sdrg_h0_hierarchy_review_v1 import dendrogram, scores  # noqa: E402
from run_sdrg_rg_scale_purity_review_v3 import DATA, get_couplings, sdrg_flow  # noqa: E402

OUT = ROOT / "results" / "review_v3" / "sdrg-cophenetic-v1"
FIGDATA = ROOT / "paper_figures/fig_sdrg_purity/data_raw/sdrg_hierarchy_box.npz"
DELTAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SIZES_LOGMEAN = (32, 64, 128, 256)
SIZES_BOX = (32, 64, 128, 256, 512)
N_REAL = 300
CUTOFF = 1e-15
TREES = ("H0", "average", "NN", "distance", "baseline")
SCORES = ("intra", "all", "auc", "within")


def nn_heights(D):
    """Cophenetic heights of single linkage restricted to nearest-neighbour links."""
    L = D.shape[0]
    d = np.diagonal(D, 1)
    H = np.zeros((L, L))
    for i in range(L - 1):
        H[i, i + 1:] = np.maximum.accumulate(d[i:])
    return H + H.T


def correlations(tree_h, join, lab, iu):
    """Scores of one tree against the SDRG hierarchy (see module docstring)."""
    t = tree_h[iu]
    j = join[iu]
    ok = np.isfinite(j)
    out = {}
    out["intra"] = float(spearmanr(t[ok], -j[ok])[0]) if ok.sum() > 2 and np.ptp(j[ok]) > 0 else np.nan
    out["all"] = float(spearmanr(t, np.where(ok, -j, 1.0))[0])
    # cluster level: probability that an intra-cluster pair is lower in the tree than an inter-cluster pair
    r = rankdata(t)
    n1, n0 = ok.sum(), (~ok).sum()
    out["auc"] = float((n1 * n0 + n1 * (n1 + 1) / 2 - r[ok].sum()) / (n1 * n0)) if n1 and n0 else np.nan
    # internal order: Spearman inside every cluster with >= 3 sites and >= 2 distinct join stages,
    # weighted by its number of pairs
    num = den = 0.0
    for k in np.unique(lab):
        sites = np.flatnonzero(lab == k)
        if sites.size < 3:
            continue
        a, b = np.triu_indices(sites.size, 1)
        x, y = tree_h[sites[a], sites[b]], -join[sites[a], sites[b]]
        if np.ptp(y) == 0 or np.ptp(x) == 0:
            continue
        w = a.size
        num += w * spearmanr(x, y)[0]
        den += w
    out["within"] = float(num / den) if den else np.nan
    return out


def task(args):
    L, seed, seed_prev, delta, ens = args
    J, G = get_couplings(L, seed, ens, delta)
    _, _, _, join = sdrg_flow(J, G)
    join_prev = sdrg_flow(*get_couplings(L, seed_prev, ens, delta))[3]
    W, _, _ = majorana_state(J, G)
    C = cstag_elimination(W, cutoff=CUTOFF)
    D = np.clip(1.0 - C, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    v = squareform(D, checks=False)
    iu = np.triu_indices(L, 1)
    heights = {"H0": squareform(cophenet(linkage(v, "single"))),
               "average": squareform(cophenet(linkage(v, "average"))),
               "NN": nn_heights(D),
               "distance": np.abs(np.subtract.outer(np.arange(L), np.arange(L))).astype(float)}
    lab = labels_from_join(join)
    out = {m: correlations(heights[m], join, lab, iu) for m in ("H0", "average", "NN", "distance")}
    out["baseline"] = correlations(heights["H0"], join_prev, labels_from_join(join_prev), iu)
    # validation: H0 dendrogram purity of the flat SDRG partition (same tree, same labels as Fig. 5)
    order = np.argsort(D[iu], kind="stable")
    nodes, merges = dendrogram(L, list(zip(iu[0][order], iu[1][order])))
    out["h0_purity"] = scores(L, lab, nodes, merges)[1]
    out["frac_pairs_joined"] = float(np.isfinite(join[iu]).mean())
    return out


def labels_from_join(join):
    L = join.shape[0]
    lab = np.full(L, -1)
    k = 0
    for i in range(L):
        if lab[i] < 0:
            lab[i] = k
            lab[np.isfinite(join[i])] = k
            k += 1
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-real", type=int, default=N_REAL)
    ap.add_argument("--ensembles", nargs="+", default=["box", "fixed_logmean"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = [("box", L, None) for L in SIZES_BOX] if "box" in a.ensembles else []
    if "fixed_logmean" in a.ensembles:
        blocks += [("fixed_logmean", L, d) for d in DELTAS for L in SIZES_LOGMEAN]
    fig = np.load(FIGDATA)
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for ens, L, delta in blocks:
            tag = f"box_L{L}" if ens == "box" else f"{ens}_d{delta:g}_L{L}"
            f = OUT / f"{tag}.npz"
            if not f.exists():
                seeds = np.load(DATA / f"random_L{L}.npz")["seeds"].astype(int)
                if ens == "box":
                    s = seeds[:a.n_real]
                    jobs = [(L, int(s[i]), int(s[i - 1]), None, ens) for i in range(s.size)]
                else:
                    s = seeds[:a.n_real + 1]
                    jobs = [(L, int(s[i]), int(s[i - 1]), delta, ens) for i in range(1, s.size)]
                t0 = time.time()
                rows = list(pool.map(task, jobs, chunksize=1 if L >= 512 else 4))
                arr = {f"{m}_{k}": np.array([r[m][k] for r in rows]) for m in TREES for k in SCORES}
                arr["h0_purity"] = np.array([r["h0_purity"] for r in rows])
                arr["frac_pairs_joined"] = np.array([r["frac_pairs_joined"] for r in rows])
                arr["seeds"] = np.array([j[1] for j in jobs])
                np.savez_compressed(f, **arr)
                print(f"{tag}: {time.time() - t0:.0f} s", flush=True)
            z = np.load(f)
            entry = {f"{m}_{k}": {"mean": float(np.nanmean(z[f"{m}_{k}"])),
                                  "sem": float(np.nanstd(z[f"{m}_{k}"]) / np.sqrt(np.isfinite(z[f"{m}_{k}"]).sum()))}
                     for m in TREES for k in SCORES}
            entry["n"] = int(z["seeds"].size)
            entry["frac_pairs_joined"] = float(z["frac_pairs_joined"].mean())
            entry["h0_purity"] = float(np.nanmean(z["h0_purity"]))
            if ens == "box":
                ref = fig[f"L{L}_H0"][:, 1]
                entry["purity_check_max_abs_dev"] = float(np.nanmax(np.abs(z["h0_purity"] - ref[:z["h0_purity"].size])))
            summary[tag] = entry
            print(tag, json.dumps({k: (round(v["mean"], 3) if isinstance(v, dict) else v) for k, v in entry.items()}), flush=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
