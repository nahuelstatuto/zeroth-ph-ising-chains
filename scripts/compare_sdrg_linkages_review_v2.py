"""Referees 1 (3.1) and 2 (3.1-3.2), round 2: is the SDRG-cluster recovery specific to H0?

For the realizations of Sec. IV B (fixed-mean, h = 0, Gamma0 = 1, seeds of
results/review_v1/free-fermion-mst-clusters-v2, 300 per size, L = 32-512) the exact correlation
matrix is recomputed and the SDRG partition (compare_sdrg_h0_clusters_review_v1.sdrg_partition:
the flow is run to the end, every site belongs to the cluster whose field decimates it, single
sites are clusters of size one) is compared with several hierarchies of D = 1 - C:
  H0        single linkage on all pairs (the H0 barcode, Kruskal on D)
  average   average linkage on all pairs (UPGMA)
  complete  complete linkage on all pairs
  NN        single linkage restricted to nearest-neighbour links (contiguous intervals)
and with a random partition of another realization (baseline, H0 tree).
Scores (scores() of compare_sdrg_h0_hierarchy_review_v1 plus two variants):
  purity        dendrogram purity, average over all pairs of sites in the same cluster (clusters
                of size >= 2 only, so a cluster of n sites weighs n(n-1)/2), as in the paper
  purity_cl     the same per cluster, then averaged over clusters with equal weight
  purity_nc3    pair-averaged purity restricted to non-contiguous clusters with at least three sites
  jaccard       best-node Jaccard, cluster-size weighted (paper)
  thr_jaccard   (H0 only) best single threshold: components of C > c for one c per realization,
                each SDRG cluster matched to its best component, size weighted, maximized over c.
                Every such component is a node of the H0 tree, so thr_jaccard <= jaccard.
Also: SDRG cluster-size distribution and fraction of non-contiguous clusters.
Run:  python scripts/compare_sdrg_linkages_review_v2.py
Output: results/review_v2/sdrg-linkages-v1/summary.json (+ per_realization.npz)
"""

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from clean_tfim_tn.disorder.reentrant import generate_disorder_base  # noqa: E402
from clean_tfim_tn.exact.free_fermion_h0 import cstag_elimination, majorana_state  # noqa: E402
from compare_sdrg_h0_clusters_review_v1 import sdrg_partition  # noqa: E402
from compare_sdrg_h0_hierarchy_review_v1 import dendrogram, scores  # noqa: E402

DATA = ROOT / "results" / "review_v1" / "free-fermion-mst-clusters-v2"
OUT = ROOT / "results" / "review_v2" / "sdrg-linkages-v1"
SIZES = (32, 64, 128, 256, 512)
N_MAX = 300
WORKERS = 12
METHODS = ("H0", "average", "complete", "NN", "baseline")


def linkage_tree(L, D, method):
    Z = linkage(squareform(D, checks=False), method=method)
    members = {i: [i] for i in range(L)}
    nodes, merges = [], []
    for t, (a, b, _, _) in enumerate(Z):
        A, B = members.pop(int(a)), members.pop(int(b))
        merges.append((A, B))
        members[L + t] = A + B
        nodes.append(A + B)
    return nodes, merges


def purity_variants(lab, merges):
    sizes = Counter(lab.tolist())
    pair_pur = Counter()
    for A, B in merges:
        ca, cb = Counter(lab[A].tolist()), Counter(lab[B].tolist())
        n = len(A) + len(B)
        for k in ca.keys() & cb.keys():
            pair_pur[k] += ca[k] * cb[k] * (ca[k] + cb[k]) / n
    per = {k: pair_pur[k] / (n * (n - 1) / 2) for k, n in sizes.items() if n >= 2}
    return per


def contiguous(idx):
    return idx.max() - idx.min() + 1 == idx.size


def thr_jaccard(L, lab, merges):
    big = {k: set(np.flatnonzero(lab == k)) for k, n in Counter(lab.tolist()).items() if n >= 2}
    comp = {i: {i} for i in range(L)}
    owner = list(range(L))
    best = 0.0
    tot = sum(len(s) for s in big.values())
    for A, B in merges:                       # merges in increasing scale = decreasing threshold c
        a, b = owner[A[0]], owner[B[0]]
        comp[a] |= comp.pop(b)
        for i in comp[a]:
            owner[i] = a
        score = 0.0
        for s in big.values():
            score += len(s) * max(len(s & comp[owner[i]]) / len(s | comp[owner[i]]) for i in s)
        best = max(best, score / tot)
    return best


def task(args):
    L, seed, seed_prev = args
    r = generate_disorder_base(L, seed).realize(1.0)
    lab = np.asarray(sdrg_partition(r.J, r.Gamma))
    rp = generate_disorder_base(L, seed_prev).realize(1.0)
    lab_prev = np.asarray(sdrg_partition(rp.J, rp.Gamma))
    C = cstag_elimination(majorana_state(r.J, r.Gamma)[0])
    D = np.clip(1.0 - C, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    iu = np.triu_indices(L, 1)
    order = np.argsort(D[iu], kind="stable")
    trees = {"H0": dendrogram(L, list(zip(iu[0][order], iu[1][order]))),
             "average": linkage_tree(L, D, "average"), "complete": linkage_tree(L, D, "complete"),
             "NN": dendrogram(L, [(i, i + 1) for i in np.argsort(np.diagonal(D, 1), kind="stable")])}
    trees["baseline"] = trees["H0"]
    sizes = Counter(lab.tolist())
    nc3 = [k for k, n in sizes.items() if n >= 3 and not contiguous(np.flatnonzero(lab == k))]
    out = {}
    for m in METHODS:
        labels = lab_prev if m == "baseline" else lab
        nodes, merges = trees[m]
        jac, pur = scores(L, labels, nodes, merges)
        per = purity_variants(labels, merges)
        sz = Counter(labels.tolist())
        if m == "baseline":
            nc = [k for k, n in sz.items() if n >= 3 and not contiguous(np.flatnonzero(labels == k))]
        else:
            nc = nc3
        w = {k: sz[k] * (sz[k] - 1) / 2 for k in nc}
        out[m] = {"jaccard": jac, "purity": pur, "purity_cl": float(np.mean(list(per.values()))) if per else np.nan,
                  "purity_nc3": float(sum(per[k] * w[k] for k in nc) / sum(w.values())) if nc else np.nan}
    out["H0"]["thr_jaccard"] = thr_jaccard(L, lab, trees["H0"][1])
    hist = Counter(min(n, 6) for n in sizes.values())
    stats = {"n_clusters": len(sizes), "size_hist": {int(k): int(v) for k, v in hist.items()},
             "sites_in_size_ge2": int(sum(n for n in sizes.values() if n >= 2)),
             "n_ge2": int(sum(1 for n in sizes.values() if n >= 2)), "n_nc3": len(nc3),
             "n_noncontig_ge2": int(sum(1 for k, n in sizes.items() if n >= 2 and not contiguous(np.flatnonzero(lab == k)))),
             "largest": int(max(sizes.values()))}
    return seed, out, stats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, per_real = {}, {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for L in SIZES:
            t0 = time.time()
            seeds = np.load(DATA / f"random_L{L}.npz")["seeds"][:N_MAX].astype(int)
            jobs = [(L, int(s), int(seeds[i - 1])) for i, s in enumerate(seeds)][1:]   # baseline: previous realization
            rows = list(pool.map(task, jobs, chunksize=2))
            res = {}
            for m in METHODS:
                for key in ("jaccard", "purity", "purity_cl", "purity_nc3"):
                    v = np.array([r[1][m][key] for r in rows], float)
                    per_real[f"L{L}_{m}_{key}"] = v
                    boot = [np.nanmean(v[np.random.default_rng(k).integers(0, v.size, v.size)]) for k in range(1000)]
                    res.setdefault(m, {})[key] = {"mean": float(np.nanmean(v)), "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                                                  "n": int(np.isfinite(v).sum())}
            tj = np.array([r[1]["H0"]["thr_jaccard"] for r in rows])
            res["H0"]["thr_jaccard"] = {"mean": float(tj.mean())}
            st = [r[2] for r in rows]
            hist = Counter()
            for s in st:
                hist.update(s["size_hist"])
            res["sdrg_stats"] = {"realizations": len(rows), "mean_clusters": float(np.mean([s["n_clusters"] for s in st])),
                                 "size_hist_total": {str(k): int(v) for k, v in sorted(hist.items())},
                                 "fraction_sites_in_size_ge2": float(np.mean([s["sites_in_size_ge2"] / L for s in st])),
                                 "mean_n_ge2": float(np.mean([s["n_ge2"] for s in st])),
                                 "mean_n_noncontig_ge2": float(np.mean([s["n_noncontig_ge2"] for s in st])),
                                 "mean_n_nc3": float(np.mean([s["n_nc3"] for s in st])),
                                 "fraction_real_with_nc3": float(np.mean([s["n_nc3"] > 0 for s in st])),
                                 "mean_largest": float(np.mean([s["largest"] for s in st]))}
            summary[f"L{L}"] = res
            print(f"L{L} {time.time() - t0:.0f} s | " + " | ".join(
                f"{m}: P {res[m]['purity']['mean']:.3f} Pcl {res[m]['purity_cl']['mean']:.3f} Pnc3 {res[m]['purity_nc3']['mean']:.3f} J {res[m]['jaccard']['mean']:.3f}"
                for m in METHODS) + f" | thrJ {res['H0']['thr_jaccard']['mean']:.3f}", flush=True)
            (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "per_realization.npz", **per_real)


if __name__ == "__main__":
    main()
