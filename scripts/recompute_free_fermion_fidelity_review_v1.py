"""Parity-consistent fidelity for the free-fermion h = 0 campaign (review v1).

The fidelity stored by run_free_fermion_h0_review_v1.py (logF) is wrong in the ordered
phase of large chains: the lowest mode falls below machine precision and consecutive
ground states can be computed in opposite parity sectors. This script recomputes
log |<psi(g_k)|psi(g_{k+1})>| with free_fermion_h0.log_fidelity_parity for every file of
the campaign (same couplings, same grid, no correlation matrices needed) and writes
fidelity_{clean,random}_L{L}.npz next to the campaign files; the campaign files are not
modified.

Run:  python scripts/recompute_free_fermion_fidelity_review_v1.py
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clean_tfim_tn.disorder.reentrant import generate_disorder_base  # noqa: E402
from clean_tfim_tn.exact.free_fermion_h0 import log_fidelity_parity, majorana_state_modes  # noqa: E402

DATA = ROOT / "results" / "review_v1" / "free-fermion-h0-v1"
WORKERS = 6


def scan_logF(J_of, grid):
    out = np.zeros(grid.size - 1)
    prev = None
    for k, g in enumerate(grid):
        W, S, u, v = majorana_state_modes(*J_of(g))
        if prev is not None:
            out[k - 1] = log_fidelity_parity(prev, W, u, v)
        prev = W
    return out


def random_task(args):
    L, seed, grid = args
    base = generate_disorder_base(L, int(seed))
    return scan_logF(lambda g: (base.realize(float(g)).J, base.realize(float(g)).Gamma), grid)


def main():
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for f in sorted(DATA.glob("clean_L*.npz")):
            if (DATA / f"fidelity_{f.stem}.npz").exists():
                continue
            z = np.load(f)
            L, grid = int(z["L"]), np.asarray(z["Gamma0"], float)
            logF = scan_logF(lambda g: (np.ones(L - 1), np.full(L, float(g))), grid)
            np.savez_compressed(DATA / f"fidelity_{f.stem}.npz", Gamma0=grid, L=L, logF=logF[None, :])
            print(f.stem, flush=True)
        for f in sorted(DATA.glob("random_L*.npz")):
            if (DATA / f"fidelity_{f.stem}.npz").exists():
                continue
            z = np.load(f)
            L, grid, seeds = int(z["L"]), np.asarray(z["Gamma0"], float), np.asarray(z["seeds"])
            logF = np.stack(list(pool.map(random_task, [(L, s, grid) for s in seeds], chunksize=8)))
            np.savez_compressed(DATA / f"fidelity_{f.stem}.npz", Gamma0=grid, L=L, seeds=seeds, logF=logF)
            print(f.stem, f"changed points: {np.mean(~np.isclose(logF, z['logF'])):.3f}", flush=True)


if __name__ == "__main__":
    main()
