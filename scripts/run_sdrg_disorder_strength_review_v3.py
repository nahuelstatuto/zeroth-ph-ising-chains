"""Referee 1 (objection 2), review v3: does the H0 hierarchy converge to the SDRG clusters
when the disorder grows and the SDRG becomes exact?

Disorder family (power law, Fisher 1995; Igloi and Monthus 2005), open chain, h = 0:
    J_i = u_i^delta ,   Gamma_i = Gamma0 v_i^delta ,   u, v uniform in (0, 1],
so that ln J and ln Gamma have standard deviation delta. The random numbers are those of
generate_disorder_base(L, seed) (u -> 1 - u, v -> 1 - v to exclude zero), and the seeds are
those of Sec. IV B (results/review_v1/free-fermion-mst-clusters-v2). The box ensemble of the
paper, J in [1, 2], has std(ln J) = 0.20, close to delta = 0.2.
Ensembles:
  fixed_logmean  (primary) the analogue of fixed_mean for this family. ln J and ln Gamma are
                 shifted so that their sample means equal the ensemble value -delta, so every
                 realization is at its own self-dual point at Gamma0 = 1.
  raw            no shift (secondary).
At Gamma0 = 1 the SDRG partition (compare_sdrg_h0_clusters_review_v1.sdrg_partition) is
compared with the hierarchies of D = 1 - C of the exact correlations, as in
compare_sdrg_linkages_review_v2: H0 (single linkage on all pairs), average, NN (single linkage
restricted to nearest-neighbour links) and the baseline (H0 tree against the SDRG partition of
the previous realization). Scores: pair-weighted dendrogram purity (paper), purity of
non-contiguous clusters with >= 3 sites, best-node Jaccard. Paired differences H0 - NN and
H0 - average with bootstrap intervals over realizations.
QC per realization: max |C| - 1, min C (Griffiths: 0 <= C <= 1), fraction of intra-cluster
SDRG pairs with C below the elimination cutoff (ties at D = 1), smallest single-particle energy.

Run (Anaconda Prompt, env tn_tda):
  python scripts/run_sdrg_disorder_strength_review_v3.py --workers 4
Output: results/review_v3/sdrg-disorder-strength-v1/{ens}_d{delta}_L{L}.npz and summary.json.
Each (ensemble, delta, L) block is saved on completion and skipped on restart.
"""

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from clean_tfim_tn.disorder.reentrant import generate_disorder_base  # noqa: E402
from clean_tfim_tn.exact.free_fermion_h0 import cstag_elimination, majorana_state  # noqa: E402
from compare_sdrg_h0_clusters_review_v1 import sdrg_partition  # noqa: E402
from compare_sdrg_h0_hierarchy_review_v1 import dendrogram, scores  # noqa: E402
from compare_sdrg_linkages_review_v2 import contiguous, linkage_tree, purity_variants  # noqa: E402

DATA = ROOT / "results" / "review_v1" / "free-fermion-mst-clusters-v2"
OUT = ROOT / "results" / "review_v3" / "sdrg-disorder-strength-v1"
DELTAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
SIZES = (32, 64, 128, 256)
ENSEMBLES = ("fixed_logmean", "raw")
N_REAL = 300
CUTOFF = 1e-15
METHODS = ("H0", "average", "NN", "baseline")
KEYS = ("purity", "purity_nc3", "jaccard")


def couplings(L, seed, delta, ensemble, Gamma0=1.0):
    b = generate_disorder_base(L, seed)
    lu, lv = np.log1p(-b.u), np.log1p(-b.v)          # ln(1 - u), finite
    if ensemble == "fixed_logmean":
        lu, lv = lu - lu.mean() - 1.0, lv - lv.mean() - 1.0
    return np.exp(delta * lu), Gamma0 * np.exp(delta * lv)


def task(args):
    L, seed, seed_prev, delta, ens = args
    J, G = couplings(L, seed, delta, ens)
    lab = np.asarray(sdrg_partition(J, G))
    lab_prev = np.asarray(sdrg_partition(*couplings(L, seed_prev, delta, ens)))
    W, S, _ = majorana_state(J, G)
    C = cstag_elimination(W, cutoff=CUTOFF)
    iu = np.triu_indices(L, 1)
    same = lab[iu[0]] == lab[iu[1]]
    qc = {"cmax_excess": float(C[iu].max() - 1.0), "cmin": float(C[iu].min()),
          "intra_tie_frac": float(np.mean(np.abs(C[iu][same]) < CUTOFF)) if same.any() else 0.0,
          "smin": float(S.min())}
    D = np.clip(1.0 - C, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    order = np.argsort(D[iu], kind="stable")
    trees = {"H0": dendrogram(L, list(zip(iu[0][order], iu[1][order]))),
             "average": linkage_tree(L, D, "average"),
             "NN": dendrogram(L, [(i, i + 1) for i in np.argsort(np.diagonal(D, 1), kind="stable")])}
    trees["baseline"] = trees["H0"]
    out = {}
    for m in METHODS:
        labels = lab_prev if m == "baseline" else lab
        nodes, merges = trees[m]
        jac, pur = scores(L, labels, nodes, merges)
        per = purity_variants(labels, merges)
        sz = Counter(labels.tolist())
        nc = [k for k, n in sz.items() if n >= 3 and not contiguous(np.flatnonzero(labels == k))]
        w = {k: sz[k] * (sz[k] - 1) / 2 for k in nc}
        out[m] = {"purity": pur, "jaccard": jac,
                  "purity_nc3": float(sum(per[k] * w[k] for k in nc) / sum(w.values())) if nc else np.nan}
    sizes = Counter(lab.tolist())
    stats = {"frac_sites_ge2": sum(n for n in sizes.values() if n >= 2) / L,
             "n_nc3": sum(1 for k, n in sizes.items() if n >= 3 and not contiguous(np.flatnonzero(lab == k))),
             "largest": max(sizes.values())}
    # fraction of H0 merges between sites that are not nearest neighbours
    nonnn, used = 0, 0
    parent = np.arange(L)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(iu[0][order], iu[1][order]):
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
            used += 1
            nonnn += abs(int(i) - int(j)) > 1
            if used == L - 1:
                break
    stats["nonnn_merges"] = nonnn / (L - 1)
    return out, stats, qc


def boot_ci(v, n=2000, seed=0):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    return np.percentile([v[rng.integers(0, v.size, v.size)].mean() for _ in range(n)], [2.5, 97.5]).tolist()


def summarize(z):
    res = {}
    for m in METHODS:
        for k in KEYS:
            v = z[f"{m}_{k}"]
            res.setdefault(m, {})[k] = {"mean": float(np.nanmean(v)), "ci95": boot_ci(v), "n": int(np.isfinite(v).sum())}
    for other in ("NN", "average"):
        d = z["H0_purity"] - z[f"{other}_purity"]
        res[f"H0_minus_{other}_purity"] = {"mean": float(np.nanmean(d)), "ci95": boot_ci(d)}
    for k in ("frac_sites_ge2", "n_nc3", "largest", "nonnn_merges"):
        res.setdefault("sdrg_stats", {})[k] = float(np.mean(z[k]))
    res["qc"] = {"cmax_excess_max": float(z["cmax_excess"].max()), "cmin_min": float(z["cmin"].min()),
                 "intra_tie_frac_mean": float(z["intra_tie_frac"].mean()),
                 "intra_tie_frac_max": float(z["intra_tie_frac"].max()),
                 "smin_min": float(z["smin"].min()), "smin_median": float(np.median(z["smin"]))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-real", type=int, default=N_REAL)
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--deltas", type=float, nargs="+", default=list(DELTAS))
    ap.add_argument("--ensembles", nargs="+", default=list(ENSEMBLES))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for ens in a.ensembles:
            for L in a.sizes:
                seeds = np.load(DATA / f"random_L{min(L, 512)}.npz")["seeds"][:a.n_real + 1].astype(int)
                for delta in a.deltas:
                    tag = f"{ens}_d{delta:g}_L{L}"
                    f = out / f"{tag}.npz"
                    if f.exists():
                        print(f"SKIP {tag}", flush=True)
                        continue
                    t0 = time.time()
                    jobs = [(L, int(seeds[i]), int(seeds[i - 1]), delta, ens) for i in range(1, seeds.size)]
                    rows = list(pool.map(task, jobs, chunksize=2))
                    arr = {f"{m}_{k}": np.array([r[0][m][k] for r in rows], float) for m in METHODS for k in KEYS}
                    arr.update({k: np.array([r[1][k] for r in rows], float) for k in rows[0][1]})
                    arr.update({k: np.array([r[2][k] for r in rows], float) for k in rows[0][2]})
                    arr["seeds"] = np.array([j[1] for j in jobs])
                    np.savez_compressed(f, **arr)
                    s = summarize(arr)
                    s["meta"] = {"ensemble": ens, "delta": delta, "L": L, "n": len(rows), "seconds": time.time() - t0}
                    summary[tag] = s
                    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                    print(f"{tag} {time.time() - t0:.0f} s | P H0 {s['H0']['purity']['mean']:.3f} "
                          f"NN {s['NN']['purity']['mean']:.3f} avg {s['average']['purity']['mean']:.3f} "
                          f"base {s['baseline']['purity']['mean']:.3f} | dP(H0-NN) {s['H0_minus_NN_purity']['mean']:+.3f} "
                          f"{s['H0_minus_NN_purity']['ci95']} | ties {s['qc']['intra_tie_frac_mean']:.2e} "
                          f"Cmax-1 {s['qc']['cmax_excess_max']:.1e} Cmin {s['qc']['cmin_min']:.1e} "
                          f"smin {s['qc']['smin_min']:.1e}", flush=True)


if __name__ == "__main__":
    main()
