"""Exploration (review v1, option 3): does the non-locality of the H0 merge tree carry physics?

Hypothesis: in the random chain at h = 0 the H0 clusters at a fixed filtration scale
(connected components of the graph C_ij >= c0, i.e. single linkage of D = 1 - C) are the
strongly correlated clusters of the strong-disorder renormalization group, which are
fractal and non-contiguous (they skip sites decimated by a strong transverse field).
At the infinite-disorder fixed point the largest cluster should grow as L^{d_f} with
d_f = phi/2 = (1 + sqrt 5)/4 = 0.809, while clusters restricted to nearest-neighbour links
(contiguous intervals) cannot follow this.

Exact free fermions (clean_tfim_tn.exact.free_fermion_h0), fixed_mean ensemble, same seeds
as the DMRG campaigns. Stored per realization and Gamma0: the full minimum spanning tree of
D (both end points and weight of the L - 1 edges), from which the clusters at any scale
follow, and the nearest-neighbour correlations C_{i,i+1}.

Run:  python scripts/run_free_fermion_mst_clusters_review_v1.py
Output: results/review_v1/free-fermion-mst-clusters-v2/random_L{L}.npz

v2: fine grid Gamma0 = 0.90-1.10 (step 0.01) for the crossing of S / L^{d_f}, and the
conventional observables for a fair comparison: m2_bulk (mean C over the central half),
C_half (mean of C_{i, i + L/2}) and C_all (mean of C over all pairs).
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clean_tfim_tn.disorder.reentrant import generate_disorder_base  # noqa: E402
from clean_tfim_tn.exact.free_fermion_h0 import cstag_elimination, majorana_state  # noqa: E402

OUT = ROOT / "results" / "review_v1" / "free-fermion-mst-clusters-v2"
WORKERS = 12
GAMMA0 = np.round(np.arange(0.90, 1.10 + 1e-9, 0.01), 4)
PLAN = {16: 1000, 32: 1000, 64: 1000, 128: 1000, 256: 1000}
GAMMA0_512 = np.array([1.0])


def task(args):
    L, seed, grid = args
    base = generate_disorder_base(L, int(seed))
    G = grid.size
    ends = np.zeros((G, L - 1, 2), np.int16)
    w = np.zeros((G, L - 1), np.float32)
    cnn = np.zeros((G, L - 1), np.float32)
    conv = np.zeros((G, 3))
    for k, g in enumerate(grid):
        r = base.realize(float(g))
        C = cstag_elimination(majorana_state(r.J, r.Gamma)[0])
        D = np.clip(1.0 - C, 0.0, 2.0) + 1.0
        np.fill_diagonal(D, 0.0)
        T = minimum_spanning_tree(np.triu(D)).tocoo()
        o = np.argsort(T.data, kind="stable")
        ends[k, :, 0], ends[k, :, 1] = T.row[o], T.col[o]
        w[k] = T.data[o] - 1.0
        cnn[k] = np.diagonal(C, 1)
        a, b = L // 4, 3 * L // 4
        conv[k] = (C[a:b, a:b].mean(), np.diagonal(C, L // 2).mean(), C[np.triu_indices(L, 1)].mean())
    return seed, ends, w, cnn, conv


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for L, n in PLAN.items():
            path = OUT / f"random_L{L}.npz"
            if path.exists():
                continue
            grid = GAMMA0_512 if L == 512 else GAMMA0
            t0 = time.time()
            res = sorted(pool.map(task, [(L, s, grid) for s in range(n)], chunksize=2), key=lambda x: x[0])
            np.savez_compressed(path, L=L, Gamma0=grid, seeds=np.array([r[0] for r in res]),
                                mst_ends=np.stack([r[1] for r in res]), mst_w=np.stack([r[2] for r in res]),
                                cnn=np.stack([r[3] for r in res]),
                                m2_bulk=np.stack([r[4][:, 0] for r in res]), C_half=np.stack([r[4][:, 1] for r in res]),
                                C_all=np.stack([r[4][:, 2] for r in res]))
            print(f"L{L}: {n} realizations, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
