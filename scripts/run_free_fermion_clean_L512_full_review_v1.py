"""Clean chain at h = 0, L = 512 on the full grid (review v1, new Fig. 2).

Exact free fermions, g = Gamma/J in [0.5, 1.5] with step 0.005 (same grid as L = 16-256 in
results/review_v1/free-fermion-h0-v1/clean_L*.npz). Per field value: sorted H0 deaths (minimum
spanning tree of D = 1 - Cstag) and the bulk squared staggered magnetization (central half).
Points are independent and computed in parallel.
Run:  python scripts/run_free_fermion_clean_L512_full_review_v1.py
Output: results/review_v1/free-fermion-h0-v1/clean_L512_full.npz
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clean_tfim_tn.exact.free_fermion_h0 import cstag_elimination, majorana_state  # noqa: E402

L = 512
GRID = np.round(np.arange(0.5, 1.5 + 1e-9, 0.005), 6)
OUT = ROOT / "results" / "review_v1" / "free-fermion-h0-v1" / "clean_L512_full.npz"


def point(g):
    C = cstag_elimination(majorana_state(np.ones(L - 1), np.full(L, float(g)))[0])
    D = np.clip(1.0 - C, 0.0, 2.0) + 1.0
    np.fill_diagonal(D, 0.0)
    T = minimum_spanning_tree(np.triu(D)).tocoo()
    a, b = L // 4, 3 * L // 4
    return np.sort(T.data - 1.0), C[a:b, a:b].mean(), float(np.mean(np.abs(T.col - T.row) == 1))


def main():
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=12) as pool:
        res = list(pool.map(point, GRID))
    np.savez_compressed(OUT, L=L, Gamma0=GRID, deaths=np.stack([r[0] for r in res]).astype(np.float32),
                        m2_bulk=np.array([r[1] for r in res]), nn_fraction=np.array([r[2] for r in res]))
    print(f"L{L}: {GRID.size} points, {time.time() - t0:.0f} s, min NN fraction {min(r[2] for r in res):.3f}", flush=True)


if __name__ == "__main__":
    main()
