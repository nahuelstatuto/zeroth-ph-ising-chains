"""Final topological review of the reentrant exit (OBC L16 h=2.5, L32 h=2.3125).

Question: can any topological observable localize the physical exit seed by
seed once the entry is excluded? Families (all computed from C=Cstag, D=1-C):

  A. H0 scalars of the Kruskal barcode (total persistence p1/p2, max/mean/std,
     quantiles, normalized entropy) and the frozen H0 landscape velocity;
  B. H1 scalars from the frozen analysis (cycle counts at several thresholds,
     total persistence);
  C. new, anatomy-motivated observables:
     - sign-skeleton percolation: largest connected component (fraction of
       sites) of the graph with edges {odd |i-j|, C_ij > 0};
     - motif coverage: fraction of half-chain pairs (i, i+L/2-1) whose interval
       contains a four-site H1 motif (square gap > 0.006).

Each scalar is searched only after the topological entry (global maximum of
the frozen H0 velocity) + 2 grid steps, with the rules: maximum, minimum,
largest positive/negative slope, largest |curvature|, and (family C) the
downward/upward crossing of 1/2. Selection on the old cohort, validation on
the new cohort, and the L16-selected rule transferred unchanged to L32.
Exploratory, no DMRG. Output: reentrant-exit-anatomy-v1/analysis/exit_topology_review.{json,csv}
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "random_ising_gamma_ensemble"
OUT = BASE / "reentrant-exit-anatomy-v1" / "analysis"
FROZEN = {16: "L16-reentrant-all1000-complete-v1", 32: "L32-reentrant-h23125-all1000-complete-v1"}
RNG = np.random.default_rng(20261014)
TAU_MOTIF = 0.006
H1_TAUS = (0.001, 0.006, 0.01, 0.02, 0.05)


def kruskal_deaths(M):
    L = M.shape[0]
    D = 1.0 - M
    iu = np.triu_indices(L, 1)
    parent = list(range(L))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    out = []
    for k in np.argsort(D[iu], kind="stable"):
        i, j = iu[0][k], iu[1][k]
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            out.append(D[i, j])
            if len(out) == L - 1:
                break
    return np.sort(out)


def largest_component_fraction(M, odd_mask):
    L = M.shape[0]
    adj = (M > 0) & odd_mask
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


def square_gaps(M):
    L = M.shape[0]
    s = np.arange(L - 3)
    return np.minimum.reduce([M[s, s + 1], M[s, s + 2], M[s + 1, s + 3], M[s + 2, s + 3]]) - np.maximum(M[s + 1, s + 2], M[s, s + 3])


def scalars(L, g, C, frozen, sel):
    """Return dict name -> (n_seeds, G) array of topological scalars."""
    n, G = len(sel), len(g)
    out = {k: np.zeros((n, G)) for k in ("H0_total_p1", "H0_total_p2", "H0_max", "H0_mean", "H0_std", "H0_q10", "H0_q50",
                                        "H0_q90", "H0_entropy", "skeleton_percolation", "motif_coverage")}
    idx = np.arange(L)
    odd = (np.abs(idx[:, None] - idx[None, :]) % 2 == 1)
    R = L // 2 - 1
    for a in range(n):
        for i in range(G):
            M = C[a, i]
            d = kruskal_deaths(M)
            p = d / d.sum()
            out["H0_total_p1"][a, i] = d.sum()
            out["H0_total_p2"][a, i] = np.sqrt((d ** 2).sum())
            out["H0_max"][a, i] = d.max()
            out["H0_mean"][a, i] = d.mean()
            out["H0_std"][a, i] = d.std()
            out["H0_q10"][a, i], out["H0_q50"][a, i], out["H0_q90"][a, i] = np.quantile(d, [0.1, 0.5, 0.9])
            out["H0_entropy"][a, i] = -(p * np.log(p + 1e-300)).sum() / np.log(len(d))
            out["skeleton_percolation"][a, i] = largest_component_fraction(M, odd)
            act = np.flatnonzero(square_gaps(M) > TAU_MOTIF)          # motif s spans sites s..s+3
            covered = [np.any((act >= start) & (act + 3 <= start + R)) for start in range(L - R)]
            out["motif_coverage"][a, i] = np.mean(covered)
    v = frozen["H0_velocity"][sel].astype(float)
    mid = frozen["Gamma0_mid"]
    out["H0_landscape_velocity"] = np.array([np.interp(g, mid, vv) for vv in v])
    th = frozen["thresholds"]
    for tau in H1_TAUS:
        out[f"H1_count_tau{tau:g}"] = frozen["H1_cycle_count"][sel][:, :, int(np.argmin(np.abs(th - tau)))].astype(float)
    if "H1_total_persistence" in frozen.files:
        out["H1_total_persistence"] = frozen["H1_total_persistence"][sel].astype(float)
    else:
        names = list(frozen["H1_scalar_names"])
        for k, name in enumerate(names):
            out[f"H1_{name}"] = frozen["H1_scalar"][sel][:, :, k].astype(float)
    return out


def locate(g, y, start, rule):
    m = g >= start
    if m.sum() < 4:
        return np.nan
    gg, yy = g[m], y[m]
    if rule == "max":
        k = int(np.argmax(yy)); return gg[k] if 0 < k < len(gg) - 1 else np.nan
    if rule == "min":
        k = int(np.argmin(yy)); return gg[k] if 0 < k < len(gg) - 1 else np.nan
    if rule in ("pos_slope", "neg_slope"):
        s = np.diff(yy) / np.diff(gg)
        k = int(np.argmax(s if rule == "pos_slope" else -s))
        return 0.5 * (gg[k] + gg[k + 1]) if np.isfinite(s[k]) and abs(s[k]) > 1e-12 else np.nan
    if rule == "abs_curv":
        c = np.abs(np.diff(yy, 2))
        k = int(np.argmax(c)); return gg[k + 1] if c[k] > 1e-12 else np.nan
    if rule in ("down_half", "up_half"):
        up = rule == "up_half"
        for k in range(len(yy) - 1):
            if (up and yy[k] < 0.5 <= yy[k + 1]) or (not up and yy[k] >= 0.5 > yy[k + 1]):
                return gg[k] + (gg[k + 1] - gg[k]) * (0.5 - yy[k]) / (yy[k + 1] - yy[k])
        return np.nan
    raise ValueError(rule)


def evaluate(est, ref, step):
    ok = np.isfinite(est)
    res = {"coverage": float(ok.mean()), "n": int(ok.sum())}
    if ok.sum() < 6:
        return res
    e, r = est[ok], ref[ok]
    mae = float(np.mean(np.abs(e - r)))
    perm = np.array([np.mean(np.abs(e - RNG.permutation(r))) for _ in range(1000)])
    res.update({"MAE": mae, "const_MAE": float(np.mean(np.abs(np.median(ref) - r))),
                "one_step": float(np.mean(np.abs(e - r) <= step + 1e-9)), "spearman": float(spearmanr(e, r).statistic),
                "perm_p": float((np.sum(perm <= mae) + 1) / 1001), "bias": float(np.mean(e - r))})
    return res


def main():
    rows, best = [], {}
    rules_default = ("max", "min", "pos_slope", "neg_slope", "abs_curv")
    for L, frozen_dir in FROZEN.items():
        cache = np.load(OUT / f"cache_L{L}.npz")
        frozen = np.load(BASE / frozen_dir / "analysis" / "analysis_arrays.npz")
        g = cache["Gamma0"]
        step = g[1] - g[0]
        sel = np.flatnonzero(cache["selected"])
        C = cache["Cstag"][sel].astype(float)
        exit_ = cache["exit"][sel]
        old = np.array([c.startswith("old") for c in cache["cohort"][sel]])
        mid = frozen["Gamma0_mid"]
        eligible = mid - step / 2 >= step - 1e-12
        top_entry = np.array([mid[eligible][np.argmax(vv[eligible])] for vv in frozen["H0_velocity"][sel].astype(float)])
        S = scalars(L, g, C, frozen, sel)
        for name, Y in S.items():
            rules = rules_default + (("down_half", "up_half") if name in ("skeleton_percolation", "motif_coverage") else ())
            for rule in rules:
                est = np.array([locate(g, Y[a], top_entry[a] + 2 * step, rule) for a in range(len(sel))])
                for subset, mask in (("all", np.ones_like(old)), ("old", old), ("new", ~old)):
                    r = evaluate(est[mask], exit_[mask], step)
                    rows.append({"L": L, "observable": name, "rule": rule, "subset": subset, **r})
                best.setdefault(L, {})[(name, rule)] = est
        print(f"L{L} done", flush=True)
    # selection on L16 old, validation L16 new, transfer to L32 (all and new)
    def get(L, subset, key):
        for r in rows:
            if r["L"] == L and r["subset"] == subset and (r["observable"], r["rule"]) == key:
                return r
    cands = [(r["observable"], r["rule"]) for r in rows if r["L"] == 16 and r["subset"] == "old" and r.get("coverage", 0) >= 0.8 and "MAE" in r]
    ranked = sorted(cands, key=lambda k: get(16, "old", k)["MAE"])
    summary = {"selection": "lowest MAE on L16 old cohort among rules with coverage >= 80%", "top10": []}
    for key in ranked[:10]:
        summary["top10"].append({"observable": key[0], "rule": key[1],
                                 "L16_old": get(16, "old", key), "L16_new": get(16, "new", key),
                                 "L32_all": get(32, "all", key), "L32_new": get(32, "new", key)})
    beats = [r for r in rows if r["subset"] == "all" and "MAE" in r and r["coverage"] >= 0.8 and r["MAE"] < r["const_MAE"]]
    summary["rules_beating_constant_all_seeds_coverage80"] = [{k: r[k] for k in ("L", "observable", "rule", "MAE", "const_MAE", "spearman")} for r in beats]
    (OUT / "exit_topology_review.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT / "exit_topology_review.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = ["L", "observable", "rule", "subset", "coverage", "n", "MAE", "const_MAE", "one_step", "spearman", "perm_p", "bias"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})
    print(json.dumps(summary, indent=1)[:6000])


if __name__ == "__main__":
    main()
