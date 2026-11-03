"""Referee 1, Sec. 3.5: is the disorder-averaged reentrance of Lajko et al. a boundary effect?

Exact diagonalization of the random chain in a longitudinal field, same fixed_mean ensemble,
same seed numbers and same distance r = L/2 - 1 for periodic (PBC, as in Lajko et al.) and open
(OBC, as in this work) chains. generate_disorder_base(L, seed, bc) draws L bonds with PBC and L - 1
with OBC, so the two boundary conditions are independent samples of the ensemble, not the same
couplings with one bond added. For each realization and Gamma0 the staggered correlator
C(r) = (-1)^r <sigma^z_i sigma^z_{i+r}> is averaged over all pairs at distance r (L pairs with
PBC, L - r with OBC), and then over realizations. Lajko et al. define the boundary by the
zero of this average.

H = sum_b J_b Z Z - Gamma0 sum_i p_i X_i - h sum_i Z_i, with J_b and p_i = Gamma_i / Gamma0
fixed along the scan in the fixed_mean ensemble. The diagonal part and the operator
X_p = sum_i p_i X_i are built once per realization (bit flips, no Kronecker products), and each
field value starts the Lanczos iteration from the previous ground state. Checked against
clean_tfim_tn.ed.random_ising.build_random_ising_hamiltonian (function check()).
Run:  python scripts/ed_reentrant_average_pbc_obc_review_v1.py
Output: results/review_v1/ed-reentrant-pbc-obc-v1/summary.json and curves.npz
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")   # one thread per worker: the pool already uses all cores

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clean_tfim_tn.disorder.reentrant import generate_disorder_base  # noqa: E402

OUT = ROOT / "results" / "review_v1" / "ed-reentrant-pbc-obc-v1"
H = 2.5
GRID = np.round(np.arange(0.0, 0.6 + 1e-9, 0.025), 4)
SIZES = (12, 16)
N_SEEDS = 300
WORKERS = 12


def operators(L, J, p, bc):
    states = np.arange(2 ** L)
    z = 1 - 2 * ((states[:, None] >> (L - 1 - np.arange(L))[None, :]) & 1)
    bonds = [(i, i + 1) for i in range(L - 1)] + ([(L - 1, 0)] if bc == "periodic" else [])
    diag = sum(Jb * z[:, a] * z[:, b] for Jb, (a, b) in zip(J, bonds)) - H * z.sum(axis=1)
    rows = np.concatenate([states] * L)
    cols = np.concatenate([states ^ (1 << (L - 1 - i)) for i in range(L)])
    vals = np.concatenate([np.full(states.size, p[i]) for i in range(L)])
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(2 ** L, 2 ** L))
    return z, diag.astype(float), X


def scan(L, seed, bc):
    base = generate_disorder_base(L, seed, bc=bc)
    J = base.realize(1.0).J
    p = base.realize(1.0).Gamma                     # Gamma_i = Gamma0 * p_i in the fixed_mean ensemble
    z, diag, X = operators(L, J, p, bc)
    r = L // 2 - 1
    pairs = [(i, (i + r) % L) for i in range(L if bc == "periodic" else L - r)]
    zz = np.stack([z[:, i] * z[:, j] for i, j in pairs], axis=1).mean(axis=1)
    out, v0 = np.zeros(GRID.size), None
    D = sparse.diags(diag)
    for k, g in enumerate(GRID):
        if g == 0.0:
            psi2 = (diag == diag.min()).astype(float)
            psi2 /= psi2.sum()
        else:
            _, v = eigsh(D - g * X, k=1, which="SA", v0=v0, tol=1e-10)
            v0 = v[:, 0]
            psi2 = v0 ** 2
        out[k] = (-1) ** r * float(psi2 @ zz)
    return L, bc, seed, out


def check():
    """Compare with the reference Hamiltonian builder at one point."""
    from clean_tfim_tn.ed.random_ising import build_random_ising_hamiltonian
    for bc in ("periodic", "open"):
        L, g = 10, 0.3
        base = generate_disorder_base(L, 5, bc=bc)
        rr = base.realize(g)
        ref = build_random_ising_hamiltonian(L, rr.J, rr.Gamma, h=H, bc=bc)
        _, diag, X = operators(L, base.realize(1.0).J, base.realize(1.0).Gamma, bc)
        mine = sparse.diags(diag) - g * X
        print(bc, "max |H_ref - H| =", abs(ref - mine).max(), flush=True)


def crossings(g, y):
    s = np.flatnonzero(np.sign(y[:-1]) != np.sign(y[1:]))
    return [float(g[i] - y[i] * (g[i + 1] - g[i]) / (y[i + 1] - y[i])) for i in s]


def main():
    check()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(L, s, bc) for L in SIZES for bc in ("periodic", "open") for s in range(N_SEEDS)]
    res, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(scan, *j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            res.append(f.result())
            if n % 100 == 0:
                print(f"{n}/{len(jobs)} scans, {time.time() - t0:.0f} s", flush=True)
    arrays, summary = {"Gamma0": GRID}, {"h": H, "seeds": N_SEEDS, "ensemble": "fixed_mean", "r": "L/2-1"}
    for L in SIZES:
        for bc in ("periodic", "open"):
            curves = np.stack([o for LL, b, s, o in sorted(res, key=lambda x: x[2]) if LL == L and b == bc])
            mean = curves.mean(0)
            sem = curves.std(0, ddof=1) / np.sqrt(len(curves))
            arrays[f"L{L}_{bc}_mean"], arrays[f"L{L}_{bc}_sem"] = mean, sem
            summary[f"L{L}_{bc}"] = {"mean_at_0": float(mean[0]), "max": float(mean.max()),
                                     "argmax": float(GRID[int(np.argmax(mean))]), "zero_crossings": crossings(GRID, mean),
                                     "sem_at_max": float(sem[int(np.argmax(mean))])}
            print(L, bc, summary[f"L{L}_{bc}"], flush=True)
    np.savez_compressed(OUT / "curves.npz", **arrays)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
