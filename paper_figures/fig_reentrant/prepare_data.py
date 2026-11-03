"""Compact, hashed input for the reentrant figures R1-R3 (OBC L16 h=2.5, L32 h=2.3125).

Built read-only from:
  results/random_ising_gamma_ensemble/reentrant-exit-anatomy-v1/analysis/cache_L{L}.npz
  results/random_ising_gamma_ensemble/reentrant-exit-anatomy-v1/analysis/local_magnetization_L{L}.npz
  frozen all-1000 analysis_arrays.npz (H0 landscape velocity, H1 cycle counts)
Only the 97 (L16) / 137 (L32) physically selected reentrant seeds are kept.
Run from the project root:  python paper_figures/fig_reentrant/prepare_data.py
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "results" / "random_ising_gamma_ensemble"
ANAT = BASE / "reentrant-exit-anatomy-v1" / "analysis"
FROZEN = {16: "L16-reentrant-all1000-complete-v1", 32: "L32-reentrant-h23125-all1000-complete-v1"}
H1_TAU = 0.006
OUT = HERE / "data_raw"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def square_gaps(M):
    L = M.shape[0]
    s = np.arange(L - 3)
    return np.minimum.reduce([M[s, s + 1], M[s, s + 2], M[s + 1, s + 3], M[s + 2, s + 3]]) - np.maximum(M[s + 1, s + 2], M[s, s + 3])


def kruskal_mean_c(M):
    """Mean C over the edges of the H0 (maximum-correlation) forest = 1 - mean H0 death."""
    L = M.shape[0]
    iu = np.triu_indices(L, 1)
    parent = list(range(L))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    chosen = []
    for k in np.argsort(-M[iu], kind="stable"):
        i, j = iu[0][k], iu[1][k]
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            chosen.append(M[i, j])
            if len(chosen) == L - 1:
                break
    return float(np.mean(chosen))


def skeleton_percolation(M):
    """Largest connected fraction of the graph {odd |i-j|, C_ij > 0}."""
    L = M.shape[0]
    idx = np.arange(L)
    adj = (M > 0) & (np.abs(idx[:, None] - idx[None, :]) % 2 == 1)
    seen, best = np.zeros(L, bool), 0
    for s in range(L):
        if seen[s]:
            continue
        stack, size = [s], 0
        seen[s] = True
        while stack:
            u = stack.pop()
            size += 1
            for v in np.flatnonzero(adj[u] & ~seen):
                seen[v] = True
                stack.append(v)
        best = max(best, size)
    return best / L


def motif_coverage(M, R, tau):
    """Fraction of windows [i, i+R] containing a four-site H1 motif (square gap > tau)."""
    L = M.shape[0]
    act = np.flatnonzero(square_gaps(M) > tau)
    return float(np.mean([np.any((act >= a) & (act + 3 <= a + R)) for a in range(L - R)]))


sys.path.insert(0, str(ROOT / "scripts"))
from analyze_obc_reentrant_selected_seed_topology import GRID, LAYERS  # noqa: E402

WIDTHS = (0, 1, 2)


def shell_forest(cstag, edges):
    """Deaths of the H0 forest of D=1-C restricted to the shell edges (frozen L16 definition)."""
    L = cstag.shape[0]
    weights = 1.0 - cstag[edges[:, 0], edges[:, 1]]
    parent = np.arange(L)
    deaths = []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for k in np.argsort(weights, kind="stable"):
        a, b = map(int, edges[k])
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            deaths.append(float(weights[k]))
    return np.sort(np.asarray(deaths))[::-1]


def long_range(C, g, L):
    """Level M_w = 1 - mean(shell forest deaths) and landscape velocity xi_LR,w for w=0,1,2."""
    R = L // 2 - 1
    step = g[1] - g[0]
    out = {}
    for w in WIDTHS:
        edges = np.asarray([(i, j) for i in range(L) for j in range(i + 1, L) if abs((j - i) - R) <= w], int)
        M = np.empty((C.shape[0], g.size))
        V = np.empty((C.shape[0], g.size - 1))
        for n in range(C.shape[0]):
            lands = np.zeros((g.size, LAYERS, GRID.size))
            for i in range(g.size):
                d = shell_forest(C[n, i], edges)
                M[n, i] = 1.0 - d.mean()
                tents = np.maximum(np.minimum(GRID[None, :], d[:, None] - GRID[None, :]), 0.0)
                ranked = np.sort(tents, axis=0)[::-1][:LAYERS]
                lands[i, :ranked.shape[0]] = ranked
            V[n] = np.sqrt(np.trapezoid(np.sum((np.diff(lands, axis=0) / step) ** 2, axis=1), GRID, axis=1) / LAYERS)
        out[f"LR_level_w{w}"] = M
        out[f"LR_velocity_w{w}"] = V
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = []
    for L, frozen_dir in FROZEN.items():
        cache_path = ANAT / f"cache_L{L}.npz"
        mag_path = ANAT / f"local_magnetization_L{L}.npz"
        frozen_path = BASE / frozen_dir / "analysis" / "analysis_arrays.npz"
        cache, mag, frozen = np.load(cache_path), np.load(mag_path), np.load(frozen_path)
        sel = np.flatnonzero(cache["selected"])
        if not np.array_equal(sel, mag["sel"]) or not np.array_equal(sel, frozen["selected_indices"]):
            raise ValueError("selection mismatch")
        g = cache["Gamma0"]
        C = cache["Cstag"][sel].astype(float)
        R = L // 2 - 1
        odd_r = np.arange(3, L - 2, 2)
        curves_r = np.stack([np.diagonal(C, offset=r, axis1=-2, axis2=-1).mean(-1) for r in odd_r], axis=1)
        pairs = np.diagonal(C, offset=R, axis1=-2, axis2=-1)                 # (n, G, L-R)
        sig = mag["mz"] * ((-1) ** np.arange(L))[None, None, :]
        th = frozen["thresholds"]
        ti = int(np.argmin(np.abs(th - H1_TAU)))
        np.savez_compressed(
            OUT / f"reentrant_L{L}.npz",
            L=L, h={16: 2.5, 32: 2.3125}[L], seeds=cache["seeds"][sel], cohort=cache["cohort"][sel],
            Gamma0=g, Gamma0_mid=frozen["Gamma0_mid"], entry=cache["entry"][sel], exit=cache["exit"][sel],
            odd_r=odd_r, curves_r=curves_r.astype(np.float32), R=R,
            pair_fraction_positive=(pairs > 0).mean(-1), pair_max=pairs.max(-1),
            H0_velocity=frozen["H0_velocity"][sel], H1_count=frozen["H1_cycle_count"][sel][:, :, ti],
            H1_tau=th[ti], staggered_abs=np.abs(sig.mean(-1)), uniform=mag["mz"].mean(-1),
            minus_dE_dGamma0=(cache["gamma_pattern"][sel][:, None, :] * mag["mx"]).sum(-1) / L,
            square_gap=np.stack([np.stack([square_gaps(C[n, i]) for i in range(g.size)]) for n in range(sel.size)]).astype(np.float32),
            J=cache["J"][sel],
            H0_backbone_level=np.array([[kruskal_mean_c(C[n, i]) for i in range(g.size)] for n in range(sel.size)]),
            skeleton_percolation=np.array([[skeleton_percolation(C[n, i]) for i in range(g.size)] for n in range(sel.size)]),
            motif_coverage=np.array([[motif_coverage(C[n, i], R, H1_TAU) for i in range(g.size)] for n in range(sel.size)]),
            curves_d=np.stack([np.diagonal(C, offset=r, axis1=-2, axis2=-1).mean(-1) for r in range(1, L)], axis=1).astype(np.float32),
            **long_range(C, g, L),
        )
        sources += [cache_path, mag_path, frozen_path]
    files = sorted(OUT.glob("reentrant_L*.npz"))
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "paper_figures/fig_reentrant/prepare_data.py",
        "conventions": "OBC; fixed_mean; C=Cstag raw (not connected); topology on D=1-C; Pauli operators",
        "selection": "frozen physical selection of the all-1000 analyses",
        "magnetization": "measured on saved MPS by scripts/measure_reentrant_local_magnetization.py (no DMRG)",
        "files": [{"file": f.name, "sha256": sha256(f), "size_bytes": f.stat().st_size} for f in files],
        "sources": [{"path": str(Path(s).relative_to(ROOT)), "sha256": sha256(s)} for s in sources],
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("ok", [f.name for f in files])


if __name__ == "__main__":
    main()
