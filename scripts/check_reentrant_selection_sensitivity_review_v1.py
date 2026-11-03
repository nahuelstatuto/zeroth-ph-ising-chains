"""Referee 2 (4.2 and Sec. 5): sensitivity of the reentrant selection to its thresholds (existing data only).

The frozen rule select_physical_seeds (Appendix A) is re-applied to the curves C(r), r = floor(L/2) - 1,
of all simulated realizations (L = 16, 32: 1000 each; L = 17, 33: 400 each), varying one threshold at a
time around the values of the paper, and with the strict set of the original analysis. For each variant:
number of selected realizations, overlap with the paper's set (Jaccard), median entry and exit, and the
realization-resolved errors of the slope of C(r) (frozen split rule) at the entry and at the exit,
compared with a constant equal to the median exit. The selected subset is also compared with the whole
sample (number of nearest-neighbour walls at Gamma0 = 0 and spread of the couplings), for the even sizes.
Run:  python scripts/check_reentrant_selection_sensitivity_review_v1.py
Output: results/review_v1/reentrant-selection-v1/summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_pbc_reentrant_selected_seed_topology import select_physical_seeds  # noqa: E402
from analyze_reentrant_review_checks import mae, split  # noqa: E402

BASE = ROOT / "results" / "random_ising_gamma_ensemble"
OUT = ROOT / "results" / "review_v1" / "reentrant-selection-v1"
PAPER = {"endpoint_points": 2, "negative_margin": 0.02, "positive_amplitude": 0.05,
         "minimum_lobe_width": 0.10, "minimum_positive_run": 2}
VARIANTS = {"positive_amplitude": (0.03, 0.08, 0.10), "minimum_lobe_width": (0.05, 0.15, 0.20),
            "negative_margin": (0.01, 0.04), "minimum_positive_run": (1, 4)}
STRICT = {**PAPER, "negative_margin": 0.04, "positive_amplitude": 0.10, "minimum_lobe_width": 0.15,
          "minimum_positive_run": 5}


def load(L):
    if L % 2 == 0:
        c = np.load(BASE / "reentrant-exit-anatomy-v1" / "analysis" / f"cache_L{L}.npz")
        C = c["Cstag"]
        R = L // 2 - 1
        curves = np.diagonal(C, offset=R, axis1=-2, axis2=-1).mean(-1).astype(float)
        walls0 = (np.diagonal(C[:, 0], offset=1, axis1=-2, axis2=-1) < -0.5).sum(-1)
        extra = {"walls0": walls0, "J": c["J"], "pattern": c["gamma_pattern"]}
        return c["Gamma0"], c["seeds"], curves, np.flatnonzero(c["selected"]), extra
    c = np.load(BASE / "reentrant-oddL-L17-L33-v1" / "analysis" / f"cache_L{L}.npz")
    return c["Gamma0"], c["seeds"], np.asarray(c["curves_R"], float), np.asarray(c["selected"], int), None


def evaluate(g, seeds, curves, params, ref_set):
    rows, sel = select_physical_seeds(g, seeds, curves, **params)
    sel = np.asarray(sel, int)
    if sel.size == 0:
        return {"n": 0}
    step = g[1] - g[0]
    mid = 0.5 * (g[1:] + g[:-1])
    entry = np.array([rows[i]["entry_crossing"] for i in sel], float)
    exit_ = np.array([rows[i]["exit_crossing"] for i in sel], float)
    slope = np.abs(np.diff(curves[sel], axis=1) / step)
    en, ex = split(slope, curves[sel], g, mid, step)
    a, b = set(sel.tolist()), set(ref_set.tolist())
    return {"n": int(sel.size), "jaccard_with_paper": len(a & b) / len(a | b),
            "median_entry": float(np.median(entry)), "median_exit": float(np.median(exit_)),
            "entry_MAE_slope": mae(en, entry, step), "exit_MAE_slope": mae(ex, exit_, step),
            "exit_MAE_constant": mae(np.full_like(exit_, np.median(exit_)), exit_, step)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"paper": PAPER, "strict": STRICT, "variants": {k: list(v) for k, v in VARIANTS.items()}}
    for L in (16, 17, 32, 33):
        g, seeds, curves, ref_sel, extra = load(L)
        _, check = select_physical_seeds(g, seeds, curves, **PAPER)
        if not np.array_equal(np.asarray(check, int), ref_sel):
            raise ValueError(f"L{L}: the frozen rule does not reproduce the stored selection")
        res = {"total": int(len(seeds)), "paper": evaluate(g, seeds, curves, PAPER, ref_sel),
               "strict": evaluate(g, seeds, curves, STRICT, ref_sel), "one_at_a_time": {}}
        for key, values in VARIANTS.items():
            for v in values:
                res["one_at_a_time"][f"{key}={v}"] = evaluate(g, seeds, curves, {**PAPER, key: v}, ref_sel)
        if extra is not None:
            s = np.zeros(len(seeds), bool)
            s[ref_sel] = True

            def stats(m):
                return {"walls0_hist": {int(k): int(n) for k, n in zip(*np.unique(extra["walls0"][m], return_counts=True))},
                        "mean_walls0": float(extra["walls0"][m].mean()),
                        "std_J": float(extra["J"][m].std(axis=1).mean()),
                        "std_gamma_pattern": float(extra["pattern"][m].std(axis=1).mean())}
            res["selected_vs_all"] = {"selected": stats(s), "not_selected": stats(~s), "all": stats(np.ones_like(s))}
        out[f"L{L}"] = res
        print(L, json.dumps({k: res[k] for k in ("total", "paper", "strict")}), flush=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
