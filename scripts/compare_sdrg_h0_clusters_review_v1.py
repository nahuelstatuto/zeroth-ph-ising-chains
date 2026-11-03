"""Do the H0 clusters of the ground-state correlations reproduce the SDRG clusters? (review v1)

For each realization (same couplings as the exact free-fermion data) we run the strong-
disorder renormalization group of the open random transverse-field Ising chain (Fisher):
at each step the largest remaining coupling Omega is decimated,
  bond J_k  -> clusters k and k+1 merge, effective field h_k h_{k+1} / J_k,
  field h_k -> cluster k is decimated (frozen along x) and recorded as a final cluster,
               neighbours joined by J_{k-1} J_k / h_k (an end cluster just drops out).
Every site ends in exactly one decimated cluster; SDRG predicts strong correlations inside
these clusters and weak ones between them. We compare this partition, realization by
realization, with
  H0   components of C_ij >= c0 (single linkage of D = 1 - C, from the stored MST),
  NN   components of the nearest-neighbour links C_{i,i+1} >= c0 (contiguous intervals),
using the adjusted Rand index (ARI, 0 for unrelated partitions, 1 for identical ones) and
the Jaccard index of the largest clusters. We also record how often the largest SDRG
cluster is non-contiguous and whether the H0 cluster recovers its far members.

Input: results/review_v1/free-fermion-mst-clusters-v2/random_L{L}.npz
Run:  python scripts/compare_sdrg_h0_clusters_review_v1.py
Output: results/review_v1/sdrg-vs-h0-v1/summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clean_tfim_tn.disorder.reentrant import generate_disorder_base  # noqa: E402

DATA = ROOT / "results" / "review_v1" / "free-fermion-mst-clusters-v2"
OUT = ROOT / "results" / "review_v1" / "sdrg-vs-h0-v1"
GAMMA0 = (0.95, 1.0, 1.05)
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def sdrg_partition(J, h):
    """Labels of the decimated SDRG cluster of every site (open chain)."""
    fields = list(map(float, h))
    bonds = list(map(float, J))
    members = [[i] for i in range(len(fields))]
    labels = np.full(len(fields), -1)
    nxt = 0
    while fields:
        kf = int(np.argmax(fields))
        kb = int(np.argmax(bonds)) if bonds else -1
        if kb >= 0 and bonds[kb] > fields[kf]:
            fields[kb] = fields[kb] * fields[kb + 1] / bonds[kb]
            members[kb] = members[kb] + members[kb + 1]
            del fields[kb + 1], members[kb + 1], bonds[kb]
        else:
            labels[members[kf]] = nxt
            nxt += 1
            n = len(fields)
            if 0 < kf < n - 1:
                bonds[kf - 1] = bonds[kf - 1] * bonds[kf] / fields[kf]
                del bonds[kf]
            elif kf == 0 and n > 1:
                del bonds[0]
            elif kf == n - 1 and n > 1:
                del bonds[kf - 1]
            del fields[kf], members[kf]
    return labels


def components(L, pairs):
    parent = np.arange(L)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(L)])


def ari(a, b):
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    M = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(M, (a, b), 1)
    comb = lambda x: x * (x - 1) / 2  # noqa: E731
    n = a.size
    s_ij, s_a, s_b = comb(M).sum(), comb(M.sum(1)).sum(), comb(M.sum(0)).sum()
    expected = s_a * s_b / comb(n)
    maximum = 0.5 * (s_a + s_b)
    return float((s_ij - expected) / (maximum - expected)) if maximum != expected else 1.0


def largest(labels):
    v, c = np.unique(labels, return_counts=True)
    return set(np.flatnonzero(labels == v[np.argmax(c)]))


def jaccard(a, b):
    return len(a & b) / len(a | b)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"Gamma0": GAMMA0, "thresholds": THRESHOLDS, "sizes": {}}
    for f in sorted(DATA.glob("random_L*.npz"), key=lambda p: int(p.stem.split("L")[1])):
        z = np.load(f)
        L, grid, seeds = int(z["L"]), np.round(np.asarray(z["Gamma0"], float), 4), z["seeds"]
        ends, w, cnn = z["mst_ends"], z["mst_w"], z["cnn"]
        res = {}
        for g0 in GAMMA0:
            if g0 not in grid:
                continue
            k = int(np.flatnonzero(grid == g0)[0])
            rows = {c0: {"ari_h0": [], "ari_nn": [], "jac_h0": [], "jac_nn": []} for c0 in THRESHOLDS}
            noncontig, far_recovered_h0, far_recovered_nn, n_clusters = 0, [], [], []
            for n, s in enumerate(seeds):
                r = generate_disorder_base(L, int(s)).realize(g0)
                sd = sdrg_partition(r.J, r.Gamma)
                n_clusters.append(len(np.unique(sd)))
                big = largest(sd)
                lo, hi = min(big), max(big)
                gaps = hi - lo + 1 - len(big)
                for c0 in THRESHOLDS:
                    h0 = components(L, [e for e, x in zip(ends[n, k], w[n, k]) if x <= 1.0 - c0])
                    nn = components(L, [(i, i + 1) for i in range(L - 1) if cnn[n, k, i] >= c0])
                    rows[c0]["ari_h0"].append(ari(sd, h0))
                    rows[c0]["ari_nn"].append(ari(sd, nn))
                    rows[c0]["jac_h0"].append(jaccard(big, largest(h0)))
                    rows[c0]["jac_nn"].append(jaccard(big, largest(nn)))
                    if c0 == 0.5 and gaps > 0:
                        # far members: the two extreme members of a non-contiguous SDRG cluster
                        far_recovered_h0.append(h0[lo] == h0[hi])
                        far_recovered_nn.append(nn[lo] == nn[hi])
                noncontig += gaps > 0
            res[str(g0)] = {
                "realizations": int(seeds.size), "sdrg_clusters_mean": float(np.mean(n_clusters)),
                "largest_sdrg_noncontiguous_fraction": noncontig / seeds.size,
                "far_members_joined_c0_0.5": {"H0": float(np.mean(far_recovered_h0)) if far_recovered_h0 else None,
                                              "NN": float(np.mean(far_recovered_nn)) if far_recovered_nn else None},
                **{f"c0={c0}": {key: float(np.mean(v)) for key, v in rows[c0].items()} for c0 in THRESHOLDS}}
            best = max(THRESHOLDS, key=lambda c0: res[str(g0)][f"c0={c0}"]["ari_h0"])
            print(f"L{L} Gamma0={g0}: SDRG clusters {np.mean(n_clusters):.1f}, largest non-contiguous {noncontig / seeds.size:.2f} | "
                  + "  ".join(f"c0={c0}: ARI H0 {res[str(g0)][f'c0={c0}']['ari_h0']:.2f} NN {res[str(g0)][f'c0={c0}']['ari_nn']:.2f}"
                              for c0 in THRESHOLDS)
                  + f" | far members joined (c0=0.5) H0 {res[str(g0)]['far_members_joined_c0_0.5']['H0']} NN {res[str(g0)]['far_members_joined_c0_0.5']['NN']}",
                  flush=True)
        summary["sizes"][f"L{L}"] = res
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
