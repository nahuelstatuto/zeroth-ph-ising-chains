"""Analysis of the H0 cluster exploration (review v1, option 3).

Input: results/review_v1/free-fermion-mst-clusters-v1/random_L{L}.npz
(run_free_fermion_mst_clusters_review_v1.py) and, for the conventional comparison,
results/review_v1/free-fermion-h0-v1/random_L{L}.npz (bulk mean correlation m2_bulk).

For each threshold c0 (cluster = component of C_ij >= c0, read from the MST edges with
weight <= 1 - c0):
  S      size of the largest cluster (number of sites)
  span   extent of the largest cluster (max - min + 1)
  fill   S / span (1 = contiguous interval; < 1 = the cluster skips sites)
  S_nn   largest contiguous run of sites linked by C_{i,i+1} >= c0 (nearest-neighbour clusters)
At Gamma0 = 1: power laws S ~ L^{d_f} on the mean and the median (L >= 32 and L >= 64),
realization bootstrap. IDFP prediction d_f = phi/2 = 0.809. Conventional comparison:
m2_bulk ~ L^{-2x}, d_f = 1 - x.

Run:  python scripts/analyze_free_fermion_mst_clusters_review_v1.py
Output: results/review_v1/free-fermion-mst-clusters-v1/analysis/summary.json
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "review_v1" / "free-fermion-mst-clusters-v1"
H0 = ROOT / "results" / "review_v1" / "free-fermion-h0-v1"
OUT = DATA / "analysis"
THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7)
DF_IDFP = (1 + 5 ** 0.5) / 4
N_BOOT = 2000
RNG = np.random.default_rng(20261018)


def clusters(L, ends, w, c0):
    parent = np.arange(L)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (i, j), x in zip(ends, w):
        if x <= 1.0 - c0:
            parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(L)])
    labels, counts = np.unique(roots, return_counts=True)
    big = labels[np.argmax(counts)]
    sites = np.flatnonzero(roots == big)
    return counts.max(), sites.max() - sites.min() + 1


def nn_run(cnn, c0):
    best = run = 1
    for c in cnn:
        run = run + 1 if c >= c0 else 1
        best = max(best, run)
    return best


def fit(Ls, y):
    m, _ = np.polyfit(np.log(Ls), np.log(y), 1)
    return float(m)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for f in sorted(DATA.glob("random_L*.npz"), key=lambda p: int(p.stem.split("L")[1])):
        z = np.load(f)
        L, grid = int(z["L"]), np.asarray(z["Gamma0"], float)
        ends, wts, cnn, nseed = z["mst_ends"], z["mst_w"], z["cnn"], z["seeds"].size   # decompress once
        out = {}
        for c0 in THRESHOLDS:
            S = np.zeros((nseed, grid.size))
            span = np.zeros_like(S)
            Snn = np.zeros_like(S)
            for n in range(S.shape[0]):
                for k in range(grid.size):
                    S[n, k], span[n, k] = clusters(L, ends[n, k], wts[n, k], c0)
                    Snn[n, k] = nn_run(cnn[n, k], c0)
            out[c0] = {"S": S, "span": span, "S_nn": Snn}
        data[L] = {"grid": grid, **out}
        print(f"L{L} done", flush=True)
    sizes = np.array(sorted(data))
    summary = {"thresholds": THRESHOLDS, "df_idfp": DF_IDFP, "at_critical": {}, "vs_Gamma0": {}, "fits": {}}
    for c0 in THRESHOLDS:
        rows = {}
        for L in sizes:
            k = int(np.argmin(np.abs(data[L]["grid"] - 1.0)))
            S, span, Snn = (data[L][c0][x][:, k] for x in ("S", "span", "S_nn"))
            rows[int(L)] = {"S_mean": float(S.mean()), "S_median": float(np.median(S)), "fill_mean": float((S / span).mean()),
                            "S_nn_mean": float(Snn.mean()), "S_over_Snn": float(S.mean() / Snn.mean())}
        summary["at_critical"][str(c0)] = rows
        for minL in (32, 64):
            use = sizes[sizes >= minL]
            if use.size < 3:
                continue
            ks = {L: int(np.argmin(np.abs(data[L]["grid"] - 1.0))) for L in use}
            samples = {L: data[L][c0]["S"][:, ks[L]] for L in use}
            nn = {L: data[L][c0]["S_nn"][:, ks[L]] for L in use}
            entry = {}
            for name, src, stat in (("S_mean", samples, np.mean), ("S_median", samples, np.median), ("S_nn_mean", nn, np.mean)):
                p = fit(use, [stat(src[L]) for L in use])
                boot = [fit(use, [stat(s[RNG.integers(0, s.size, s.size)]) for s in (src[L] for L in use)]) for _ in range(N_BOOT)]
                entry[name] = {"d_f": p, "ci95": np.percentile(boot, [2.5, 97.5]).tolist()}
            summary["fits"][f"c0={c0}_Lmin{minL}"] = entry
            print(f"c0={c0} L>={minL}: d_f(mean) {entry['S_mean']['d_f']:.3f} {np.round(entry['S_mean']['ci95'], 3)}  "
                  f"d_f(median) {entry['S_median']['d_f']:.3f}  NN-run exponent {entry['S_nn_mean']['d_f']:.3f}", flush=True)
    for c0 in THRESHOLDS:
        summary["vs_Gamma0"][str(c0)] = {int(L): {"Gamma0": data[L]["grid"].tolist(),
                                                  "S_over_L": (data[L][c0]["S"].mean(0) / L).tolist(),
                                                  "S_nn_over_L": (data[L][c0]["S_nn"].mean(0) / L).tolist(),
                                                  "fill": (data[L][c0]["S"] / data[L][c0]["span"]).mean(0).tolist()}
                                         for L in sizes if data[L]["grid"].size > 1}
    # conventional route: bulk mean correlation at Gamma0 = 1 from the h = 0 campaign
    conv = {}
    for L in sizes:
        f = H0 / f"random_L{L}.npz"
        if f.exists():
            z = np.load(f)
            k = int(np.argmin(np.abs(z["Gamma0"] - 1.0)))
            conv[int(L)] = float(z["m2_bulk"][:, k].mean())
    if len(conv) >= 3:
        Ls = np.array(sorted(L for L in conv if L >= 32))
        two_x = -fit(Ls, [conv[L] for L in Ls])
        summary["conventional"] = {"m2_bulk_mean": conv, "two_x": two_x, "d_f_from_m2": 1 - two_x / 2}
        print(f"conventional: m2_bulk ~ L^-{two_x:.3f}, d_f = 1 - x = {1 - two_x / 2:.3f} (IDFP {DF_IDFP:.3f})", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
