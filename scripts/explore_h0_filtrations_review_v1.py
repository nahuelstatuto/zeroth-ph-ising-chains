"""Referee test (review v1): does the H0 merge tree stay nearest-neighbour for other complexes?

For H0, every filtration in which vertices are born at 0 and an edge enters at a
monotone function of D_ij (Vietoris-Rips, Cech/alpha on the exact embedding, Dowker,
lazy witness with nu = 0) has the single-linkage merge tree, i.e. the minimum
spanning tree (MST) of D. C = Cstag is a correlation (Gram) matrix with unit
diagonal, so x_i with |x_i - x_j|^2 = 2 (1 - C_ij) is an exact Euclidean embedding and
the Cech/alpha H0 is the Euclidean MST of that embedding, a monotone reparametrization
of the Rips one. We check this numerically and compare with filtrations that change
the merge tree:

  rips        Vietoris-Rips on D = 1 - C (paper)
  cech        Rips on |x_i - x_j| / 2 = sqrt(2 D) / 2 (Cech/alpha H0 of the exact embedding)
  witness2    lazy witness complex, nu = 2, all sites landmarks and witnesses
              (edge at min_w max(D_wi, D_wj) - m_w, m_w = distance from w to its 2nd landmark)
  dtm3, dtm5  DTM-Rips (GUDHI, q = 2, k = 3 and 5): vertices born at the distance to measure
  profile     Rips on the correlation-profile metric d_ij = rms_l (C_il - C_jl) (other metric)

For each filtration and field value: H0 pairs (elder rule), fraction of merging edges
between nearest neighbours, fraction replaceable by the nearest-neighbour chain path at
the same value (tie-robust), landscape velocity (central differences, all layers) and its
maximum (parabola refinement).

Data (no new DMRG): clean h = 0, L = 16-128 (paper_figures/fig_clean_h0/data_raw);
random h = 0, L = 32 and 64, 50 realizations each (Gamma-critical campaigns, coarse grid).
Run with the tn_tda environment (GUDHI):
    python scripts/explore_h0_filtrations_review_v1.py
Output: results/review_v1/h0-filtrations-v1/{summary.json, curves.npz}
"""

import glob
import json
import re
from pathlib import Path

import numpy as np
from gudhi.dtm_rips_complex import DTMRipsComplex

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "review_v1" / "h0-filtrations-v1"
CLEAN = {16: 64, 32: 128, 64: 128, 128: 256}
RANDOM = {32: "random-h0-Gamma-critical-L32-c128-s0-49-v1", 64: "random-h0-Gamma-critical-L64-c128-s0-49-v1"}
FILTRATIONS = ("rips", "cech", "witness2", "dtm3", "dtm5", "profile")
GRID_N = 801


def h0_pairs(birth, edge):
    """Elder-rule H0 pairs from vertex births and a symmetric edge-value matrix."""
    L = birth.size
    iu, ju = np.triu_indices(L, 1)
    val = np.maximum(edge[iu, ju], np.maximum(birth[iu], birth[ju]))
    order = np.lexsort((ju - iu, val))
    parent = np.arange(L)
    root_birth = birth.copy()

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs, edges = [], []
    for k in order:
        a, b = find(iu[k]), find(ju[k])
        if a == b:
            continue
        young, old = (a, b) if root_birth[a] > root_birth[b] or (root_birth[a] == root_birth[b] and a > b) else (b, a)
        pairs.append((root_birth[young], val[k]))
        edges.append((iu[k], ju[k], val[k]))
        parent[young] = old
    return np.array(pairs), edges


def nn_stats(edges, edge, birth):
    """Fraction of merging edges that are nearest neighbours, and tie-robust fraction."""
    nn = rep = 0
    for i, j, v in edges:
        if j - i == 1:
            nn += 1
            rep += 1
            continue
        path = max(max(edge[k, k + 1], birth[k], birth[k + 1]) for k in range(i, j))
        rep += path <= v + 1e-12
    return nn / len(edges), rep / len(edges)


def filtration(C, name):
    L = C.shape[0]
    D = np.clip(1.0 - C, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    zero = np.zeros(L)
    if name == "rips":
        return zero, D
    if name == "cech":
        return zero, np.sqrt(2.0 * D) / 2.0
    if name == "witness2":
        m = np.sort(D, axis=1)[:, 1]                      # distance from w to its 2nd landmark (1st is itself)
        E = np.maximum(D[:, :, None], D[:, None, :]) - m[:, None, None]
        E = np.maximum(E.min(axis=0), 0.0)
        np.fill_diagonal(E, 0.0)
        return zero, E
    if name.startswith("dtm"):
        st = DTMRipsComplex(distance_matrix=D, k=int(name[3:]), q=2).create_simplex_tree(max_dimension=1)
        b = np.zeros(L)
        E = np.zeros((L, L))
        for simplex, f in st.get_filtration():
            if len(simplex) == 1:
                b[simplex[0]] = f
            else:
                E[simplex[0], simplex[1]] = E[simplex[1], simplex[0]] = f
        return b, E
    if name == "profile":
        P = np.sqrt(np.mean((C[:, None, :] - C[None, :, :]) ** 2, axis=-1))
        np.fill_diagonal(P, 0.0)
        return zero, P
    raise ValueError(name)


def scan(Cs, g):
    """Per filtration: NN fractions per field, landscape velocity and its maximum."""
    out = {}
    for name in FILTRATIONS:
        diagrams, fnn, frep = [], [], []
        for C in Cs:
            b, E = filtration(C, name)
            pairs, edges = h0_pairs(b, E)
            diagrams.append(pairs)
            a, r = nn_stats(edges, E, b)
            fnn.append(a)
            frep.append(r)
        top = max(float(p[:, 1].max()) for p in diagrams)
        grid = np.linspace(0.0, top, GRID_N)
        lands = np.stack([np.sort(np.maximum(np.minimum(grid[None, :] - p[:, :1], p[:, 1:] - grid[None, :]), 0.0),
                                  axis=0)[::-1] for p in diagrams])
        dl = np.gradient(lands, g, axis=0)
        xi = np.sqrt(np.trapezoid(np.sum(dl ** 2, axis=1), grid, axis=1) / lands.shape[1])
        out[name] = {"nn": np.array(fnn), "rep": np.array(frep), "xi": xi, "peak": peak(g, xi)}
    return out


def peak(x, y):
    i = int(np.argmax(y[1:-1])) + 1
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    return float(x[i] + (0.5 * (a - c) / den * (x[i + 1] - x[i]) if den != 0 else 0.0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, arrays = {"filtrations": FILTRATIONS, "clean_h0": {}, "random_h0": {}}, {}
    for L, chi in CLEAN.items():
        z = np.load(ROOT / "paper_figures" / "fig_clean_h0" / "data_raw" / f"clean_h0_L{L}_chi{chi}.npz")
        g, C = z["Gamma_values"] / float(z["J"]), z["Cstag"]
        min_eig = float(min(np.linalg.eigvalsh(c).min() for c in C))
        res = scan(C, g)
        summary["clean_h0"][f"L{L}"] = {"min_eigenvalue_C": min_eig, **{
            n: {"peak": r["peak"], "nn_min": float(r["nn"].min()), "nn_mean": float(r["nn"].mean()),
                "rep_min": float(r["rep"].min()),
                "xi_rel_diff_vs_rips": float(np.max(np.abs(r["xi"] / r["xi"].max() - res["rips"]["xi"] / res["rips"]["xi"].max())))}
            for n, r in res.items()}}
        for n, r in res.items():
            arrays[f"clean_L{L}_{n}_xi"] = r["xi"]
            arrays[f"clean_L{L}_{n}_nn"] = r["nn"]
        arrays[f"clean_L{L}_g"] = g
        print("clean", L, {n: round(r["peak"], 4) for n, r in res.items()}, flush=True)
    for L, camp in RANDOM.items():
        peaks, nn, rep, seeds = {n: [] for n in FILTRATIONS}, {n: [] for n in FILTRATIONS}, {n: [] for n in FILTRATIONS}, []
        for f in sorted(glob.glob(str(ROOT / "results" / "random_ising_gamma_ensemble" / camp / "scans" / "*" / "continuation_summary.npz"))):
            z = np.load(f)
            res = scan(z["Cstag"], z["Gamma0_values"])
            seeds.append(int(re.search(r"seed(\d+)_", f).group(1)))
            for n, r in res.items():
                peaks[n].append(r["peak"])
                nn[n].append(r["nn"].mean())
                rep[n].append(r["rep"].mean())
        summary["random_h0"][f"L{L}"] = {n: {
            "peak_mean": float(np.mean(peaks[n])), "peak_std": float(np.std(peaks[n])),
            "median_abs_diff_vs_rips": float(np.median(np.abs(np.array(peaks[n]) - np.array(peaks["rips"])))),
            "identical_peak_vs_rips": int(np.sum(np.abs(np.array(peaks[n]) - np.array(peaks["rips"])) < 1e-9)),
            "nn_mean": float(np.mean(nn[n])), "rep_mean": float(np.mean(rep[n]))} for n in FILTRATIONS}
        summary["random_h0"][f"L{L}"]["realizations"] = len(seeds)
        for n in FILTRATIONS:
            arrays[f"random_L{L}_{n}_peaks"] = np.array(peaks[n])
        arrays[f"random_L{L}_seeds"] = np.array(seeds)
        print("random", L, {n: round(v["peak_std"], 4) for n, v in summary["random_h0"][f"L{L}"].items() if n in FILTRATIONS}, flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "curves.npz", **arrays)


if __name__ == "__main__":
    main()
