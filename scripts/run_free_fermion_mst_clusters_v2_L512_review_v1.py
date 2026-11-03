"""L = 512 extension of the refined H0 cluster campaign (review v1, option 3).

Same observables as run_free_fermion_mst_clusters_v2_review_v1.py, narrow window
Gamma0 = 0.96-1.04 (step 0.01) and 300 realizations (seeds 0-299), to test whether the
crossing of <S>/L^{d_f} at Gamma0 = 1 holds up to L = 512.
Run:  python scripts/run_free_fermion_mst_clusters_v2_L512_review_v1.py
Output: results/review_v1/free-fermion-mst-clusters-v2/random_L512.npz
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_free_fermion_mst_clusters_v2_review_v1 as v2  # noqa: E402

L, N = 512, 300
GRID = np.round(np.arange(0.96, 1.04 + 1e-9, 0.01), 4)


def main():
    path = v2.OUT / f"random_L{L}.npz"
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=v2.WORKERS) as pool:
        res = sorted(pool.map(v2.task, [(L, s, GRID) for s in range(N)], chunksize=1), key=lambda x: x[0])
    np.savez_compressed(path, L=L, Gamma0=GRID, seeds=np.array([r[0] for r in res]),
                        mst_ends=np.stack([r[1] for r in res]), mst_w=np.stack([r[2] for r in res]),
                        cnn=np.stack([r[3] for r in res]),
                        m2_bulk=np.stack([r[4][:, 0] for r in res]), C_half=np.stack([r[4][:, 1] for r in res]),
                        C_all=np.stack([r[4][:, 2] for r in res]))
    print(f"L{L}: {N} realizations, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
