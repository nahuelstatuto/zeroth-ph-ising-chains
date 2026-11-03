"""Free-fermion (exact) h = 0 campaign for the referee benchmark (review v1).

Open chains, H = sum J_i Z_i Z_{i+1} - sum Gamma_i X_i, fixed_mean ensemble with the
same disorder generator and seeds as the DMRG campaigns (generate_disorder_base(L, seed)),
so seeds 0-49 reproduce the DMRG realizations exactly. C = Cstag raw, topology on
D = 1 - C. No DMRG: clean_tfim_tn.exact.free_fermion_h0 (validated against ED to 1e-14
and against the DMRG matrices to <= 1e-8 on unflagged points).

Stored per realization and field value (for every estimator of the benchmark):
  deaths   sorted finite H0 deaths (MST weights of D), float32 (G, L-1)
  sep      |i - j| of the MST edge of each sorted death, int16 (G, L-1)
  cnn      nearest-neighbour correlations C_{i,i+1}, float32 (G, L-1)
  energy   ground-state energy (G)
  m2_bulk  squared staggered magnetization of the central half, mean of C over B x B (G)
  S_half   half-chain entanglement entropy (G)
  logF     log fidelity between consecutive field values (G - 1)
  dC_fro   ||C(g_{k+1}) - C(g_{k-1})||_F / (2 dg L), central differences (G, NaN at the ends)

Run (tn_tda or base Python with numpy/scipy):
    python scripts/run_free_fermion_h0_review_v1.py
Output: results/review_v1/free-fermion-h0-v1/{clean,random}_L{L}.npz + manifest.json
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clean_tfim_tn.disorder.reentrant import generate_disorder_base  # noqa: E402
from clean_tfim_tn.exact.free_fermion_h0 import cstag_elimination, log_fidelity, majorana_state  # noqa: E402

OUT = ROOT / "results" / "review_v1" / "free-fermion-h0-v1"
WORKERS = 12
FULL = np.round(np.arange(0.5, 1.5 + 1e-9, 0.005), 6)
RANDOM = {8: (1000, FULL), 16: (1000, FULL), 32: (1000, FULL), 64: (1000, FULL), 128: (1000, FULL),
          256: (500, np.round(np.arange(0.7, 1.3 + 1e-9, 0.005), 6))}
CLEAN = {16: FULL, 32: FULL, 64: FULL, 128: FULL, 256: FULL, 512: np.round(np.arange(0.9, 1.1 + 1e-9, 0.0025), 6)}


def entropy_half(W):
    L = W.shape[0]
    nu = np.clip(np.linalg.svd(W[: L // 2, : L // 2], compute_uv=False), 0.0, 1.0)
    p = 0.5 * (1.0 + nu)
    q = 1.0 - p
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(np.where(p > 0, p * np.log(p), 0.0) + np.where(q > 0, q * np.log(q), 0.0))
    return float(h.sum())


def scan(J_of, grid, L):
    G = grid.size
    out = {"deaths": np.zeros((G, L - 1), np.float32), "sep": np.zeros((G, L - 1), np.int16),
           "cnn": np.zeros((G, L - 1), np.float32), "energy": np.zeros(G), "m2_bulk": np.zeros(G),
           "S_half": np.zeros(G), "logF": np.zeros(G - 1), "dC_fro": np.full(G, np.nan)}
    B = slice(L // 4, L // 4 + L // 2)
    Cs, W_prev = [], None
    for k, g in enumerate(grid):
        J, Gam = J_of(g)
        W, S, E = majorana_state(J, Gam)
        C = cstag_elimination(W)
        D = np.clip(1.0 - C, 0.0, 2.0) + 1.0          # unit offset: zero distances stay edges
        np.fill_diagonal(D, 0.0)
        T = minimum_spanning_tree(np.triu(D)).tocoo()
        w = np.asarray(T.data) - 1.0
        o = np.argsort(w, kind="stable")
        out["deaths"][k] = w[o]
        out["sep"][k] = np.abs(T.col - T.row)[o]
        out["cnn"][k] = np.diagonal(C, 1)
        out["energy"][k] = E
        out["m2_bulk"][k] = C[B, B].mean()
        out["S_half"][k] = entropy_half(W)
        if W_prev is not None:
            out["logF"][k - 1] = log_fidelity(W_prev, W)
        W_prev = W
        Cs.append(C)
        if len(Cs) == 3:
            out["dC_fro"][k - 1] = np.linalg.norm(Cs[2] - Cs[0]) / (grid[k] - grid[k - 2]) / L
            Cs.pop(0)
    return out


def random_task(args):
    L, seed, grid = args
    base = generate_disorder_base(L, seed)

    def J_of(g):
        r = base.realize(float(g))
        return r.J, r.Gamma

    return seed, scan(J_of, grid, L)


def clean_task(args):
    L, grid = args
    return L, scan(lambda g: (np.ones(L - 1), np.full(L, float(g))), grid, L)


def save(path, grid, results, extra):
    keys = results[0].keys()
    np.savez_compressed(path, Gamma0=grid, **extra, **{k: np.stack([r[k] for r in results]) for k in keys})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "generator": "scripts/run_free_fermion_h0_review_v1.py",
                "conventions": "OBC; fixed_mean; C=Cstag raw; topology on D=1-C; Pauli operators; h=0 exact free fermions",
                "files": []}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        t0 = time.time()
        for L, res in pool.map(clean_task, [(L, g) for L, g in CLEAN.items()]):
            path = OUT / f"clean_L{L}.npz"
            save(path, CLEAN[L], [res], {"L": L})
            manifest["files"].append({"file": path.name, "L": L, "realizations": 1, "grid": [float(CLEAN[L][0]), float(CLEAN[L][-1]), int(CLEAN[L].size)]})
            print(f"clean L{L} done {time.time() - t0:.0f} s", flush=True)
        for L, (n, grid) in RANDOM.items():
            t0 = time.time()
            out = sorted(pool.map(random_task, [(L, s, grid) for s in range(n)], chunksize=4), key=lambda x: x[0])
            path = OUT / f"random_L{L}.npz"
            save(path, grid, [r for _, r in out], {"L": L, "seeds": np.array([s for s, _ in out])})
            manifest["files"].append({"file": path.name, "L": L, "realizations": n, "grid": [float(grid[0]), float(grid[-1]), int(grid.size)]})
            print(f"random L{L}: {n} realizations, {time.time() - t0:.0f} s", flush=True)
            (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
