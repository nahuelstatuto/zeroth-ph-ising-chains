"""Exploratory scan of H1 estimator parameters on the clean OBC odd-L cuts.

Not a frozen analysis: it maps how the H1 proliferation estimate depends on the
lifetime threshold tau, on the minimum jump dN1 required to define it, on whether
the end-closure ring is removed, and on the peak rule (largest jump or first jump).
Any parameter chosen here by agreement with the frontier is a fit and has to be
validated on data not used to choose it (see the leave-one-cut-out block).

Uses the complete cuts of clean-L{31,63}-oddL-fullgamma-h*-c128-v1. Bars are cached
per scan (lifetime, ring flag, Gamma index). Output:
results/clean_ising_gamma_campaign/clean-frontier-oddL-closure-v1/exploration/
"""

from __future__ import annotations

import csv
import itertools
import json

import numpy as np

from analyze_clean_frontier_oddL_closure import (
    BASE, CHI, FRONTIER, H_GRID, HIT, SIZES, complete_scan, distance, h1_bars, sha256,
)

OUTPUT = BASE / "clean-frontier-oddL-closure-v1" / "exploration"
TAUS = (0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060)
JUMPS = (1, 2, 3, 4, 5, 6)
COUNTS = ("bulk", "all")       # ring removed / ring included
RULES = ("largest", "first")   # largest jump >= jmin / first interval with jump >= jmin


def bars_for(path):
    cache = OUTPUT / "cache" / f"{path.parent.name}_{sha256(path)[:16]}_bars.npz"
    if cache.exists():
        with np.load(cache) as z:
            return {k: np.asarray(z[k]) for k in z.files}
    with np.load(path, allow_pickle=False) as z:
        gamma = np.asarray(z["Gamma_values"], float)
        cstag = np.asarray(z["Cstag"], float)
        bond = np.asarray(z["selected_max_bond_dimension"], int)
        disagree = np.asarray(z["branch_disagreement"], bool)
    idx, life, ring = [], [], []
    for g in range(gamma.size):
        for lt, rg in h1_bars(distance(cstag[g]), ring_rule=True):
            idx.append(g), life.append(lt), ring.append(rg)
    out = {"gamma": gamma, "invalid": disagree | (bond >= CHI), "idx": np.array(idx, int),
           "life": np.array(life, float), "ring": np.array(ring, bool)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def n1(b, tau, count):
    keep = b["life"] > tau
    if count == "bulk":
        keep &= ~b["ring"]
    return np.bincount(b["idx"][keep], minlength=b["gamma"].size)


def estimate(b, tau, jmin, count, rule):
    gamma, n = b["gamma"], n1(b, tau, count)
    jump = np.diff(n)
    valid = ~(b["invalid"][:-1] | b["invalid"][1:]) & (jump >= jmin)
    if not valid.any():
        return None
    if rule == "first":
        i = int(np.flatnonzero(valid)[0])
    else:
        i = int(np.argmax(np.where(valid, jump, -1)))  # first on ties (uniform grid)
    return float(0.5 * (gamma[i] + gamma[i + 1]))


def score(cuts, params):
    errs, missing = [], 0
    for (L, h), b in cuts.items():
        e = estimate(b, *params)
        if e is None:
            missing += 1
        else:
            errs.append(abs(e - FRONTIER[h]))
    errs = np.array(errs)
    return {"n_est": int(errs.size), "n_none": missing,
            "mae": float(errs.mean()) if errs.size else None,
            "hits": int(np.sum(errs <= HIT)), "max_err": float(errs.max()) if errs.size else None}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cuts = {}
    for L in SIZES:
        for h in H_GRID:
            path = complete_scan(L, h)
            if path is not None:
                print(f"bars L={L} h={h}", flush=True)
                cuts[(L, h)] = bars_for(path)
    grid = list(itertools.product(TAUS, JUMPS, COUNTS, RULES))
    rows = []
    for params in grid:
        s = score(cuts, params)
        per = {f"L{L}_h{h}": estimate(b, *params) for (L, h), b in sorted(cuts.items())}
        rows.append({"tau": params[0], "jmin": params[1], "count": params[2], "rule": params[3], **s, **per})
    with (OUTPUT / "parameter_scan.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Leave-one-cut-out: choose parameters on the other cuts (most hits, then lowest MAE,
    # requiring an estimate on every training cut), predict the held-out cut.
    loco = []
    keys = sorted(cuts)
    for held in keys:
        train = {k: v for k, v in cuts.items() if k != held}
        best = None
        for params in grid:
            s = score(train, params)
            if s["n_none"] or s["mae"] is None:
                continue
            key = (-s["hits"], s["mae"])
            if best is None or key < best[0]:
                best = (key, params)
        if best is None:
            loco.append({"held": f"L{held[0]}_h{held[1]}", "params": None})
            continue
        e = estimate(cuts[held], *best[1])
        loco.append({"held": f"L{held[0]}_h{held[1]}", "params": best[1], "estimate": e,
                     "frontier": FRONTIER[held[1]],
                     "error": None if e is None else abs(e - FRONTIER[held[1]])})
    summary = {"cuts": [f"L{L}_h{h}" for L, h in keys], "n_params": len(grid),
               "top_by_hits_then_mae": sorted(
                   [r for r in rows if r["n_none"] == 0],
                   key=lambda r: (-r["hits"], r["mae"]))[:15],
               "leave_one_cut_out": loco}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
