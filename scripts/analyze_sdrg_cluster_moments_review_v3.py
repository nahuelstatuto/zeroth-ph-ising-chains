"""B3 follow-up (review v3, R1-2): why the H0 tree does not follow the order of the SDRG decimations
inside a cluster (analyze_sdrg_cophenetic_review_v3), and why the NN tree does.

Inside every final SDRG cluster with >= 4 sites (pair-weighted averages per realization):
  r2_rank1      R^2 of the least-squares fit ln C_ij = u_i + u_j (factorization into site moments
                mu_i = e^{u_i}, C_ij ~ mu_i mu_j)
  h0_weak       Spearman between the H0 cophenetic height of (i, j) and -min(u_i, u_j): the tree joins
                the sites in the order of their moments
  h0_join       Spearman between the H0 height and the SDRG join order (as `within` of the main script)
  partial_nn    partial Spearman of the NN height and the SDRG join order given |i - j|
  partial_h0    same for the H0 height
Ensembles and seeds as analyze_sdrg_cophenetic_review_v3, L = 256 (box and fixed_logmean deltas).
Run (tn_tda):  python scripts/analyze_sdrg_cluster_moments_review_v3.py --workers 12
Output: results/review_v3/sdrg-cophenetic-v1/cluster_moments_L256.json
"""

import argparse
import json
import os
import sys
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
from analyze_sdrg_cophenetic_review_v3 import CUTOFF, DATA, OUT, labels_from_join, nn_heights  # noqa: E402
from run_sdrg_rg_scale_purity_review_v3 import get_couplings, sdrg_flow  # noqa: E402

L = 256
N_REAL = 300
ENSEMBLES = [("box", None)] + [("fixed_logmean", d) for d in (0.25, 0.5, 1.0, 2.0)]
KEYS = ("r2_rank1", "h0_weak", "h0_join", "partial_nn", "partial_h0")


def partial(x, y, z):
    r = np.corrcoef([rankdata(x), rankdata(y), rankdata(z)])
    return (r[0, 1] - r[0, 2] * r[1, 2]) / np.sqrt((1 - r[0, 2] ** 2) * (1 - r[1, 2] ** 2))


def task(args):
    seed, ens, delta = args
    J, G = get_couplings(L, seed, ens, delta)
    join = sdrg_flow(J, G)[3]
    lab = labels_from_join(join)
    W, _, _ = majorana_state(J, G)
    C = cstag_elimination(W, cutoff=CUTOFF)
    D = np.clip(1.0 - C, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    H0 = squareform(cophenet(linkage(squareform(D, checks=False), "single")))
    NN = nn_heights(D)
    acc = {k: [0.0, 0.0] for k in KEYS}
    for k in np.unique(lab):
        st = np.flatnonzero(lab == k)
        if st.size < 4:
            continue
        a, b = np.triu_indices(st.size, 1)
        y = np.log(np.clip(C[st[a], st[b]], 1e-300, None))
        A = np.zeros((a.size, st.size))
        A[np.arange(a.size), a] = 1.0
        A[np.arange(a.size), b] = 1.0
        u = np.linalg.lstsq(A, y, rcond=None)[0]
        jo = -join[st[a], st[b]]
        dist = np.abs(st[a] - st[b]).astype(float)
        h0, nn = H0[st[a], st[b]], NN[st[a], st[b]]
        vals = {"r2_rank1": 1 - np.sum((y - A @ u) ** 2) / np.sum((y - y.mean()) ** 2) if np.ptp(y) > 0 else np.nan,
                "h0_weak": spearmanr(h0, -np.minimum(u[a], u[b]))[0],
                "h0_join": spearmanr(h0, jo)[0] if np.ptp(jo) > 0 else np.nan,
                "partial_nn": partial(nn, jo, dist) if np.ptp(jo) > 0 else np.nan,
                "partial_h0": partial(h0, jo, dist) if np.ptp(jo) > 0 else np.nan}
        for key, v in vals.items():
            if np.isfinite(v):
                acc[key][0] += a.size * v
                acc[key][1] += a.size
    return {k: (n / w if w else np.nan) for k, (n, w) in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    seeds = np.load(DATA / f"random_L{L}.npz")["seeds"].astype(int)
    out = {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for ens, delta in ENSEMBLES:
            s = seeds[:N_REAL] if ens == "box" else seeds[1:N_REAL + 1]
            rows = list(pool.map(task, [(int(x), ens, delta) for x in s], chunksize=4))
            tag = "box" if ens == "box" else f"d{delta:g}"
            out[tag] = {k: {"mean": float(np.nanmean([r[k] for r in rows])),
                            "sem": float(np.nanstd([r[k] for r in rows]) / np.sqrt(len(rows)))} for k in KEYS}
            out[tag]["n"] = len(rows)
            print(tag, {k: round(v["mean"], 3) for k, v in out[tag].items() if k != "n"}, flush=True)
    (OUT / f"cluster_moments_L{L}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
