"""Modularity cut of the H0 hierarchy against the SDRG clusters (review v3, Hemery et al. 2022).

Hemery, Pollmann and Smith [PRB 105, 064202 (2022)] remove the weakest bonds of the correlation
graph one by one and keep the partition of maximal modularity. Their sequence of partitions is
the single-linkage hierarchy, i.e. the H0 merge tree read from the top. Here that cut is applied
to the realizations of Sec. IV B (fixed-mean, h = 0, Gamma0 = 1, the 300 seeds per size of
compare_sdrg_linkages_review_v2, L = 32-512) and scored against the SDRG partition with the same
size-weighted Jaccard as the best single threshold and the best subtrees of the paper:
  weights      M_ij = C_ij (i != j), the similarity of D = 1 - C
  modularity   Q = (1/2m) sum_ij (M_ij - k_i k_j / 2m) delta(c_i, c_j), updated at every merge
               of the H0 tree (merges in increasing D), the partition of largest Q is kept
  jaccard      each SDRG cluster with >= 2 sites matched to its best block, size weighted
Also stored: Q_max, the number of blocks, and the fraction of merges done at the cut.

Multiscale extraction without reference (fixed before looking at the scores): HDBSCAN [Campello,
Moulavi and Sander 2013] with min_samples = 1, so that the reachability distance is D itself and
the condensed tree is the H0 single-linkage tree, min_cluster_size = 2, excess-of-mass selection.
It keeps the clusters of largest stability (summed persistence of their sites) at any scale, and
the sites left as noise are singletons. Two scales for the persistence, the same tree:
  hdbscan_D      D = 1 - C, the filtration of the paper (primary)
  hdbscan_logC   -ln C, the logarithmic scale of the renormalization group (secondary)
Run (tn_tda):  python scripts/sdrg_modularity_cut_review_v3.py
Output: results/review_v3/sdrg-modularity-cut-v1/summary.json (+ per_realization.npz)
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
from sklearn.cluster import HDBSCAN

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from clean_tfim_tn.disorder.reentrant import generate_disorder_base  # noqa: E402
from clean_tfim_tn.exact.free_fermion_h0 import cstag_elimination, majorana_state  # noqa: E402
from compare_sdrg_h0_clusters_review_v1 import sdrg_partition  # noqa: E402
from compare_sdrg_linkages_review_v2 import DATA, N_MAX, SIZES  # noqa: E402

OUT = ROOT / "results" / "review_v3" / "sdrg-modularity-cut-v1"
WORKERS = int(os.environ.get("WORKERS", "2"))
SIZES = tuple(int(x) for x in os.environ["SIZES"].split(",")) if "SIZES" in os.environ else SIZES


def modularity_cut(L, C):
    """Partition of maximal modularity along the H0 merges, with M = C off the diagonal."""
    M = np.clip(C, 0.0, None)
    np.fill_diagonal(M, 0.0)
    k = M.sum(1)
    two_m = k.sum()
    D = 1.0 - C
    iu = np.triu_indices(L, 1)
    order = np.argsort(D[iu], kind="stable")
    owner = np.arange(L)
    blocks = {i: [i] for i in range(L)}
    K = {i: k[i] for i in range(L)}
    Q = -(k ** 2).sum() / two_m ** 2                 # all sites apart
    best = (Q, {i: list(b) for i, b in blocks.items()}, 0)
    merges = 0
    for i, j in zip(iu[0][order], iu[1][order]):
        a, b = owner[i], owner[j]
        if a == b:
            continue
        A, B = blocks[a], blocks.pop(b)
        Q += 2.0 * (M[np.ix_(A, B)].sum() / two_m - K[a] * K[b] / two_m ** 2)
        A.extend(B)
        owner[B] = a
        K[a] += K.pop(b)
        merges += 1
        if Q > best[0]:
            best = (Q, {x: list(y) for x, y in blocks.items()}, merges)
        if merges == L - 1:
            break
    labels = np.empty(L, int)
    for n, members in enumerate(best[1].values()):
        labels[members] = n
    return labels, float(best[0]), best[2]


def hdbscan_partition(dist):
    """HDBSCAN on a precomputed distance, noise sites as singletons."""
    lab = HDBSCAN(metric="precomputed", min_cluster_size=2, min_samples=1,
                  cluster_selection_method="eom", copy=True).fit_predict(dist)
    noise = lab < 0
    lab[noise] = lab.max() + 1 + np.arange(noise.sum())
    return lab


def jaccard(lab_sdrg, lab_cut):
    big = {c: set(np.flatnonzero(lab_sdrg == c)) for c, n in Counter(lab_sdrg.tolist()).items() if n >= 2}
    blocks = {c: set(np.flatnonzero(lab_cut == c)) for c in np.unique(lab_cut)}
    num = sum(len(s) * max(len(s & b) / len(s | b) for b in blocks.values()) for s in big.values())
    return num / sum(len(s) for s in big.values())


def task(args):
    L, seed = args
    r = generate_disorder_base(L, seed).realize(1.0)
    lab = np.asarray(sdrg_partition(r.J, r.Gamma))
    C = cstag_elimination(majorana_state(r.J, r.Gamma)[0])
    cut, q, merges = modularity_cut(L, C)
    out = {"jaccard": jaccard(lab, cut), "Q": q, "n_blocks": int(np.unique(cut).size),
           "n_sdrg": int(np.unique(lab).size), "merge_fraction": merges / (L - 1)}
    D = np.clip(1.0 - C, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    logC = -np.log(np.clip(C, 1e-300, 1.0))
    np.fill_diagonal(logC, 0.0)
    for name, dist in (("hdbscan_D", D), ("hdbscan_logC", logC)):
        h = hdbscan_partition(dist)
        out[f"jaccard_{name}"] = jaccard(lab, h)
        out[f"n_blocks_{name}"] = int(np.unique(h).size)
        out[f"n_multi_{name}"] = int(sum(n >= 2 for n in Counter(h.tolist()).values()))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prev = json.loads((ROOT / "results" / "review_v2" / "sdrg-linkages-v1" / "summary.json").read_text(encoding="utf-8"))
    sp, pp = OUT / "summary.json", OUT / "per_realization.npz"       # sizes run separately are merged
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    per = dict(np.load(pp)) if pp.exists() else {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for L in SIZES:
            t0 = time.time()
            seeds = np.load(DATA / f"random_L{L}.npz")["seeds"][:N_MAX].astype(int)[1:]   # as in v2 (no baseline)
            rows = list(pool.map(task, [(L, int(s)) for s in seeds], chunksize=2))
            res = {}
            for key in rows[0]:
                v = np.array([r_[key] for r_ in rows], float)
                per[f"L{L}_{key}"] = v
                boot = [v[np.random.default_rng(b).integers(0, v.size, v.size)].mean() for b in range(1000)]
                res[key] = {"mean": float(v.mean()), "ci95": np.percentile(boot, [2.5, 97.5]).tolist()}
            res["n"] = len(rows)
            res["reference_v2"] = {"best_subtree_jaccard": prev[f"L{L}"]["H0"]["jaccard"]["mean"],
                                   "best_single_threshold_jaccard": prev[f"L{L}"]["H0"]["thr_jaccard"]["mean"]}
            summary[f"L{L}"] = res
            print(f"L{L} {time.time() - t0:.0f} s | modularity cut J {res['jaccard']['mean']:.3f} "
                  f"Q {res['Q']['mean']:.3f} blocks {res['n_blocks']['mean']:.1f} (SDRG {res['n_sdrg']['mean']:.1f}) | "
                  f"HDBSCAN D J {res['jaccard_hdbscan_D']['mean']:.3f} logC J {res['jaccard_hdbscan_logC']['mean']:.3f} | "
                  f"threshold J {res['reference_v2']['best_single_threshold_jaccard']:.3f} "
                  f"subtrees J {res['reference_v2']['best_subtree_jaccard']:.3f}", flush=True)
            sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            np.savez_compressed(pp, **per)


if __name__ == "__main__":
    main()
