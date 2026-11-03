"""Search for a single H1 lifetime threshold tau that works for the clean frontier and the reentrant exit.

Exploratory (user request 2026-10-16). Two H1 rules:
  first_rise : midpoint of the first QC-valid interval where N1(tau) increases
               (clean frontier rule, amendment v2, all bars);
  max_growth : midpoint of the interval with the largest increase of N1(tau)
               (rule of the reentrant figure).
For the reentrant cohorts (L16, L17, L32, L33, physically selected seeds) the rules are
applied to the fields after the physical entry, and compared with the physical exit and
with a constant equal to the median exit. For the clean frontier (L31, L63, h = 1.1-1.9)
the first-rise rule is compared with the digitized boundary (cached bars of
explore_clean_frontier_oddL_h1_params.py).
Output: results/random_ising_gamma_ensemble/h1-tau-common-v1/{lifetimes_L*.npz, summary.json}
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_clean_L64_new_topology import h1_diagram  # noqa: E402

BASE = ROOT / "results" / "random_ising_gamma_ensemble"
OUT = BASE / "h1-tau-common-v1"
TAUS = np.array([0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08])


def lifetimes_even(L):
    cache = np.load(BASE / "reentrant-exit-anatomy-v1" / "analysis" / f"cache_L{L}.npz")
    sel = np.flatnonzero(cache["selected"])
    return cache["Gamma0"], cache["Cstag"][sel].astype(float), cache["entry"][sel], cache["exit"][sel]


def lifetimes_odd(L):
    cache = np.load(BASE / "reentrant-oddL-L17-L33-v1" / "analysis" / f"cache_L{L}.npz")
    d = np.load(ROOT / "paper_figures" / "fig_reentrant" / "data_raw" / f"reentrant_L{L}.npz")
    seeds = cache["seeds"][cache["selected"]]
    prefix = {17: "L17-reentrant-h2500", 33: "L33-reentrant-h23125"}[L]
    dirs = {}
    for st in BASE.glob(f"{prefix}-oddL-*/campaign_status.json"):
        for j in json.loads(st.read_text(encoding="utf-8"))["jobs"]:
            dirs[int(j["seed"])] = Path(j["target_directory"])
    C = np.stack([np.load(dirs[int(s)] / "continuation_summary.npz")["Cstag"].astype(float) for s in seeds])
    return cache["Gamma0"], C, d["entry"], d["exit"]


def compute(L):
    path = OUT / f"lifetimes_L{L}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["g"], list(z["life"]), z["entry"], z["exit"]
    g, C, entry, exit_ = (lifetimes_even if L % 2 == 0 else lifetimes_odd)(L)
    life = []
    for n in range(C.shape[0]):
        per = []
        for i in range(g.size):
            dia = h1_diagram(C[n, i])
            per.append(dia[:, 1] - dia[:, 0] if dia.size else np.zeros(0))
        life.append(per)
    obj = np.empty(len(life), dtype=object)
    for k, v in enumerate(life):
        inner = np.empty(len(v), dtype=object)
        for i, x in enumerate(v):
            inner[i] = x
        obj[k] = inner
    np.savez_compressed(path, g=g, life=obj, entry=entry, exit=exit_)
    return g, list(obj), entry, exit_


def estimate(g, per, tau, rule, start):
    n1 = np.array([int(np.sum(x > tau)) for x in per])
    jump = np.diff(n1)
    ok = (0.5 * (g[1:] + g[:-1]) > start) & (jump >= 1)
    if not ok.any():
        return np.nan
    i = int(np.flatnonzero(ok)[0]) if rule == "first_rise" else int(np.argmax(np.where(ok, jump, -1)))
    return float(0.5 * (g[i] + g[i + 1]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"reentrant": {}, "clean": {}}
    for L in (16, 17, 32, 33):
        g, life, entry, exit_ = compute(L)
        const = float(np.mean(np.abs(np.median(exit_) - exit_)))
        res = {"constant_MAE": const, "N": len(life)}
        for rule in ("first_rise", "max_growth"):
            for tau in TAUS:
                est = np.array([estimate(g, per, tau, rule, e) for per, e in zip(life, entry)])
                ok = np.isfinite(est)
                res[f"{rule}_tau{tau:g}"] = {"MAE": float(np.mean(np.abs(est[ok] - exit_[ok]))) if ok.any() else None,
                                             "N": int(ok.sum()), "median": float(np.nanmedian(est)) if ok.any() else None}
        out["reentrant"][f"L{L}"] = res
        print(L, "done", flush=True)
    # clean frontier: first rise on the cached odd-L bars
    from explore_clean_frontier_oddL_h1_params import bars_for, estimate as clean_est
    from analyze_clean_frontier_oddL_closure import complete_scan, FRONTIER, H_GRID
    for tau in TAUS:
        errs = {}
        for L in (31, 63):
            e = []
            for h in H_GRID:
                v = clean_est(bars_for(complete_scan(L, h)), float(tau), 1, "all", "first")
                e.append(np.nan if v is None else abs(v - FRONTIER[h]))
            errs[f"L{L}"] = {"MAE": float(np.nanmean(e)), "hits": int(np.sum(np.array(e) <= 0.05))}
        out["clean"][f"tau{tau:g}"] = errs
    (OUT / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
