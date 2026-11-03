"""Frozen transfer of the L16 long-range H0 observables to OBC L32 (h=2.3125).

Rules fixed in documentation/RESULTADOS_L16_TOPOLOGIA_LARGA_DISTANCIA_20261012.md
before looking at L32:
  primary     : maximum positive slope of the polarization P, relative shell w=2
                (L16 w=1 -> L32 w=2, r = L/2-1 = 15);
  sensitivity : same with w=4;
  unified     : long-range landscape velocity of the same shell, entry = its
                maximum to the left of the apex of the level M;
  bridge      : w=0 (identity with the physical correlator; not a validation).
Definitions are copied from scripts/analyze_L16_long_range_topological_observables.py
(forest deaths on shell edges, P = (mean deaths(D-) - mean deaths(D+))/2, first
eligible interval excluded with FIRST_STATE = 2*STEP). Inputs: the L32 cache of
reentrant-exit-anatomy-v1 (all 1000 seeds) and the frozen selection.
Output: reentrant-exit-anatomy-v1/analysis/L32_long_range_transfer.{json,csv}
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_obc_reentrant_selected_seed_topology import GRID, LAYERS  # noqa: E402

BASE = ROOT / "results" / "random_ising_gamma_ensemble"
OUT = BASE / "reentrant-exit-anatomy-v1" / "analysis"
FROZEN = BASE / "L32-reentrant-h23125-all1000-complete-v1" / "analysis" / "analysis_arrays.npz"
L = 32
R = L // 2 - 1
WIDTHS = (0, 2, 4)
N_BOOT = 5000
RNG = np.random.default_rng(20261014)


def edges_for_width(width):
    return np.asarray([(i, j) for i in range(L) for j in range(i + 1, L) if abs((j - i) - R) <= width], int)


def forest_deaths(cstag, edges, sign=1.0):
    weights = 1.0 - sign * cstag[edges[:, 0], edges[:, 1]]
    parent = np.arange(L)
    rank = np.zeros(L, np.int8)
    deaths = []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for k in np.argsort(weights, kind="stable"):
        a, b = map(int, edges[k])
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        deaths.append(float(weights[k]))
    return np.sort(np.asarray(deaths, float))[::-1]


def landscape_from_deaths(deaths):
    result = np.zeros((LAYERS, GRID.size), np.float32)
    tents = np.maximum(np.minimum(GRID[None, :], deaths[:, None] - GRID[None, :]), 0.0)
    ranked = np.sort(tents, axis=0)[::-1]
    result[:min(LAYERS, deaths.size)] = ranked[:LAYERS]
    return result


def metrics(pred, target, step):
    valid = np.isfinite(pred) & np.isfinite(target)
    p, t = pred[valid], target[valid]
    if p.size < 3:
        return {"N": int(p.size), "coverage": float(p.size / target.size)}
    ae = np.abs(p - t)
    draws = RNG.integers(0, p.size, size=(N_BOOT, p.size))
    perm = np.array([np.mean(np.abs(p - RNG.permutation(t))) for _ in range(2000)])
    return {"N": int(p.size), "coverage": float(p.size / target.size), "MAE": float(ae.mean()),
            "MAE_ci95": [float(v) for v in np.quantile(ae[draws].mean(axis=1), [0.025, 0.975])],
            "median_AE": float(np.median(ae)), "bias": float(np.mean(p - t)),
            "within_step": int(np.sum(ae <= step + 1e-12)), "spearman": float(spearmanr(p, t).statistic),
            "perm_p": float((np.sum(perm <= ae.mean()) + 1) / 2001),
            "constant_median_MAE": float(np.mean(np.abs(np.median(t) - t)))}


def paired_difference(a, b, target):
    """MAE(a) - MAE(b) on seeds where both are finite, bootstrap CI."""
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(target)
    da, db = np.abs(a[ok] - target[ok]), np.abs(b[ok] - target[ok])
    draws = RNG.integers(0, da.size, size=(N_BOOT, da.size))
    diffs = da[draws].mean(1) - db[draws].mean(1)
    return {"N": int(ok.sum()), "delta_MAE": float(da.mean() - db.mean()),
            "ci95": [float(v) for v in np.quantile(diffs, [0.025, 0.975])]}


def main():
    cache = np.load(OUT / "cache_L32.npz")
    frozen = np.load(FROZEN)
    g = cache["Gamma0"]
    step = float(g[1] - g[0])
    mid = 0.5 * (g[1:] + g[:-1])
    first_state = 2 * step
    eligible_mid = mid >= first_state + step / 2 - 1e-12
    C_all = cache["Cstag"].astype(float)
    n_all = C_all.shape[0]
    sel = np.flatnonzero(cache["selected"])
    entry = cache["entry"][sel]
    new = np.array([not c.startswith("old") for c in cache["cohort"][sel]])
    sign_class = cache["sign_class"]

    curves = {}
    for w in WIDTHS:
        edges = edges_for_width(w)
        M = np.empty((n_all, g.size))
        P = np.empty((n_all, g.size))
        V = np.empty((n_all, g.size - 1))
        for s in range(n_all):
            lands = np.empty((g.size, LAYERS, GRID.size), np.float32)
            for i in range(g.size):
                plus = forest_deaths(C_all[s, i], edges, +1.0)
                minus = forest_deaths(C_all[s, i], edges, -1.0)
                M[s, i] = 1.0 - plus.mean()
                P[s, i] = (minus.mean() - plus.mean()) / 2.0
                lands[i] = landscape_from_deaths(plus)
            deriv = np.diff(lands.astype(float), axis=0) / step
            V[s] = np.sqrt(np.trapezoid(np.sum(deriv ** 2, axis=1), GRID, axis=1) / LAYERS)
        curves[w] = (M, P, V)
        print(f"w={w} done", flush=True)

    def slope_entry(P):
        slope = np.diff(P, axis=1) / step
        idx = np.flatnonzero(eligible_mid)
        return mid[idx[np.argmax(slope[:, idx], axis=1)]]

    def split_entry(V, M):
        out = np.full(V.shape[0], np.nan)
        state_ok = g >= first_state - 1e-12
        for s in range(V.shape[0]):
            apex = g[state_ok][np.argmax(M[s, state_ok])]
            left = np.flatnonzero(eligible_mid & (mid <= apex + 1e-12))
            if left.size:
                out[s] = mid[left[np.argmax(V[s, left])]]
        return out

    def crossing_entry(curve):
        for i in range(g.size - 1):
            y0, y1 = curve[i], curve[i + 1]
            if y0 <= 0 < y1:
                return g[i] - y0 * step / (y1 - y0)
        return math.nan

    v_frozen = frozen["H0_velocity"].astype(float)
    elig_frozen = frozen["Gamma0_mid"] - step / 2 >= step - 1e-12
    global_velocity = np.array([frozen["Gamma0_mid"][elig_frozen][np.argmax(vv[elig_frozen])] for vv in v_frozen])

    detectors = {}
    for w in WIDTHS:
        M, P, V = curves[w]
        detectors[f"P_slope_w{w}"] = slope_entry(P)
        detectors[f"LR_landscape_split_w{w}"] = split_entry(V, M)
    detectors["global_landscape_velocity"] = global_velocity

    summary = {"protocol": {"L": L, "h": 2.3125, "r": R, "widths": WIDTHS, "step": step, "first_state": first_state,
                            "rules": "frozen in RESULTADOS_L16_TOPOLOGIA_LARGA_DISTANCIA_20261012.md"},
               "entry": {}, "paired_vs_global_velocity": {}, "false_entry_in_always_negative": {}}
    rows = []
    for name, est_all in detectors.items():
        est = est_all[sel]
        for subset, mask in (("all137", np.ones_like(new)), ("old37", ~new), ("new100", new)):
            m = metrics(est[mask], entry[mask], step)
            summary["entry"].setdefault(name, {})[subset] = m
            rows.append({"detector": name, "subset": subset, **{k: v for k, v in m.items() if not isinstance(v, list)}})
        if name != "global_landscape_velocity":
            summary["paired_vs_global_velocity"][name] = {
                subset: paired_difference(est[mask], global_velocity[sel][mask], entry[mask])
                for subset, mask in (("all137", np.ones_like(new)), ("new100", new))}
    always_negative = np.flatnonzero(sign_class == "always_negative")
    for w in WIDTHS:
        M, P, _ = curves[w]
        false = np.array([np.isfinite(crossing_entry(P[s])) for s in always_negative])
        summary["false_entry_in_always_negative"][f"P_w{w}"] = {"count": int(false.sum()), "of": int(always_negative.size)}
    (OUT / "L32_long_range_transfer.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT / "L32_long_range_transfer.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k != "detector", k != "subset", k)))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(OUT / "L32_long_range_curves.npz", **{f"{k}_w{w}": curves[w][i] for w in WIDTHS for i, k in enumerate("MPV")})
    for name, d in summary["entry"].items():
        a = d["all137"]
        print(f"{name:28s} all137 MAE {a.get('MAE', float('nan')):.4f} {a.get('within_step')}/{a['N']} rho {a.get('spearman', float('nan')):.3f} | "
              f"old37 {d['old37'].get('MAE', float('nan')):.4f} | new100 {d['new100'].get('MAE', float('nan')):.4f}")
    print(json.dumps(summary["paired_vs_global_velocity"], indent=1))
    print(summary["false_entry_in_always_negative"])


if __name__ == "__main__":
    main()
