"""Threshold-free comparison of the H0 hierarchy with the SDRG clusters (review v1).

A fixed cut C_ij >= c0 of the single-linkage dendrogram needs a threshold and chains
distinct clusters in large chains. SDRG clusters are not defined at one scale: each is
decimated at its own energy scale. We therefore compare the whole H0 dendrogram (all merge
scales of the barcode) with the SDRG partition, without any threshold:

  best-node Jaccard  for every SDRG cluster with at least two sites, the best Jaccard index
                     over all subtrees (nodes) of the dendrogram, averaged with weights equal
                     to the cluster size;
  dendrogram purity  (Heller and Ghahramani 2005) average over pairs of sites in the same
                     SDRG cluster of the fraction of that cluster in the smallest subtree
                     containing both.

Dendrograms: H0 (minimum spanning tree of D, stored) and NN (single linkage restricted to
nearest-neighbour links, i.e. contiguous intervals). Baseline: the same scores for the SDRG
partition of an independent realization (unrelated clusters).

Input: results/review_v1/free-fermion-mst-clusters-v2/random_L{L}.npz at Gamma0 = 1.
Run:  python scripts/compare_sdrg_h0_hierarchy_review_v1.py
Output: results/review_v1/sdrg-vs-h0-v1/hierarchy_summary.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from clean_tfim_tn.disorder.reentrant import generate_disorder_base  # noqa: E402
from compare_sdrg_h0_clusters_review_v1 import DATA, OUT, sdrg_partition  # noqa: E402

N_MAX = 300


def dendrogram(L, edges):
    """Merge sequence of sorted edges -> list of node member arrays and the merge pairs."""
    parent = np.arange(L)
    members = {i: [i] for i in range(L)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    nodes, merges = [], []
    for i, j in edges:
        a, b = find(i), find(j)
        if a == b:
            continue
        merges.append((list(members[a]), list(members[b])))
        parent[a] = b
        members[b] = members[b] + members.pop(a)
        nodes.append(members[b])
    return nodes, merges


def scores(L, labels, nodes, merges):
    lab = np.asarray(labels)
    sizes = Counter(lab.tolist())
    # best-node Jaccard
    num = den = 0.0
    big = {k: set(np.flatnonzero(lab == k)) for k, n in sizes.items() if n >= 2}
    node_sets = [set(n) for n in nodes]
    for k, s in big.items():
        best = max(len(s & ns) / len(s | ns) for ns in node_sets)
        num += len(s) * best
        den += len(s)
    jac = num / den if den else np.nan
    # dendrogram purity
    tot = pur = 0.0
    for A, B in merges:
        ca, cb = Counter(lab[A].tolist()), Counter(lab[B].tolist())
        n = len(A) + len(B)
        for k in ca.keys() & cb.keys():
            pairs = ca[k] * cb[k]
            pur += pairs * (ca[k] + cb[k]) / n
            tot += pairs
    return jac, (pur / tot if tot else np.nan)


def main():
    out, per = {}, {}
    for f in sorted(DATA.glob("random_L*.npz"), key=lambda p: int(p.stem.split("L")[1])):
        z = np.load(f)
        L = int(z["L"])
        if L < 32:
            continue
        grid = np.round(np.asarray(z["Gamma0"], float), 4)
        k = int(np.flatnonzero(grid == 1.0)[0])
        ends, cnn, seeds = z["mst_ends"], z["cnn"], z["seeds"]
        n_use = min(N_MAX, seeds.size)
        res = {"H0": [], "NN": [], "H0_baseline": []}
        sd_prev = None
        for n in range(n_use):
            r = generate_disorder_base(L, int(seeds[n])).realize(1.0)
            sd = sdrg_partition(r.J, r.Gamma)
            h0_nodes, h0_merges = dendrogram(L, [tuple(e) for e in ends[n, k]])        # stored in increasing weight
            order = np.argsort(1.0 - cnn[n, k], kind="stable")
            nn_nodes, nn_merges = dendrogram(L, [(i, i + 1) for i in order])
            res["H0"].append(scores(L, sd, h0_nodes, h0_merges))
            res["NN"].append(scores(L, sd, nn_nodes, nn_merges))
            if sd_prev is not None:
                res["H0_baseline"].append(scores(L, sd_prev, h0_nodes, h0_merges))
            sd_prev = sd
        per[f"L{L}"] = res
        out[f"L{L}"] = {key: {"best_node_jaccard": float(np.nanmean([v[0] for v in vals])),
                              "dendrogram_purity": float(np.nanmean([v[1] for v in vals])), "n": len(vals)}
                        for key, vals in res.items()}
        o = out[f"L{L}"]
        print(f"L{L}: best-node Jaccard H0 {o['H0']['best_node_jaccard']:.3f} NN {o['NN']['best_node_jaccard']:.3f} "
              f"baseline {o['H0_baseline']['best_node_jaccard']:.3f} | purity H0 {o['H0']['dendrogram_purity']:.3f} "
              f"NN {o['NN']['dendrogram_purity']:.3f} baseline {o['H0_baseline']['dendrogram_purity']:.3f}", flush=True)
    (OUT / "hierarchy_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "hierarchy_per_realization.npz", **{f"{L}_{key}": np.array(v) for L, d in per.items() for key, v in d.items()})


if __name__ == "__main__":
    main()
