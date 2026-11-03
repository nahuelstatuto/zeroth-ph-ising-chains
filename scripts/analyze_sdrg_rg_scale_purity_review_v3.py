"""Analysis of run_sdrg_rg_scale_purity_review_v3 (results/review_v3/sdrg-rg-scale-v1).

0. Purity per joining stage (main). Every intra-cluster pair is binned by the stage rho at which
   the SDRG joined it (octaves down to 2^-9, rho_edges of the run). Per bin: pair purity of the
   H0, average and NN trees, 95% bootstrap interval over realizations, share of the pairs.
   The bins add up to the global purity of the paper.
1. Purity per RG stage of the cluster. Final SDRG clusters (size >= 2) are binned by rho_dec, the fraction of
   active clusters when they are decimated (early ~ 1, late -> 0), in octaves
   (1/2, 1], (1/4, 1/2], ..., (0, 1/32]. Per bin: pair-weighted purity of the H0, average and
   NN trees (sum of pair purities / number of pairs, as the global purity of the paper),
   95% bootstrap interval over realizations, number of clusters and share of the pair weight.
2. Stopped flow (B2): mean purity of the partition stopped at rho_stop, all clusters and active
   clusters only, for H0, average, NN and the baseline.
3. Checks: fixed_logmean global purities against C1 (sdrg-disorder-strength-v1, same seeds),
   box against review_v2/sdrg-linkages-v1 (our seeds[1:300] are its 299 realizations).
Output: analysis/summary.json and the exploratory figure analysis/purity_by_rg_stage.{png,pdf}.
Run:  python scripts/analyze_sdrg_rg_scale_purity_review_v3.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "review_v3" / "sdrg-rg-scale-v1"
C1 = ROOT / "results" / "review_v3" / "sdrg-disorder-strength-v1"
V2 = ROOT / "results" / "review_v2" / "sdrg-linkages-v1" / "per_realization.npz"
OUT = RUN / "analysis"
EDGES = 2.0 ** -np.arange(0, 6)[::-1]          # 1/32 ... 1
EDGES = np.concatenate([[0.0], EDGES])          # bins (0,1/32], (1/32,1/16], ..., (1/2,1]
TREES = ("H0", "average", "NN")
RHO_STOPS = (0.5, 0.25, 0.125, 0.0625)
N_BOOT = 1000


def strata(z):
    size = z["cl_size"]
    big = size >= 2
    w = size * (size - 1) / 2
    real = z["cl_real"].astype(int)
    nreal = z["seeds"].size
    rho = z["cl_rho_dec"]
    out = []
    wtot = w[big].sum()
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, nreal, nreal) for _ in range(N_BOOT)]
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = big & (rho > lo) & (rho <= hi)
        row = {"rho_lo": float(lo), "rho_hi": float(hi), "n_clusters": int(sel.sum()),
               "n_real_with": int(np.unique(real[sel]).size), "pair_weight_share": float(w[sel].sum() / wtot),
               "mean_size": float(size[sel].mean()) if sel.any() else None,
               "frac_noncontig": float(np.mean(z["cl_contiguous"][sel] == 0)) if sel.any() else None}
        for m in TREES:
            if not sel.any():
                row[m] = None
                continue
            num = np.bincount(real[sel], weights=z[f"cl_pur_{m}"][sel] * w[sel], minlength=nreal)
            den = np.bincount(real[sel], weights=w[sel], minlength=nreal)
            bs = [num[b].sum() / den[b].sum() for b in boots if den[b].sum() > 0]
            row[m] = {"purity": float(num.sum() / den.sum()), "ci95": np.percentile(bs, [2.5, 97.5]).tolist()}
        out.append(row)
    return out


def pair_strata(z):
    edges = z["rho_edges"]
    den = z["pairbin_den"]
    nreal = den.shape[0]
    rng = np.random.default_rng(1)
    boots = [rng.integers(0, nreal, nreal) for _ in range(N_BOOT)]
    out = []
    for i in range(den.shape[1]):
        if den[:, i].sum() == 0:
            continue
        row = {"rho_lo": float(edges[i]), "rho_hi": float(edges[i + 1]), "pairs": float(den[:, i].sum()),
               "pair_share": float(den[:, i].sum() / den.sum()), "n_real_with": int((den[:, i] > 0).sum())}
        for m in TREES:
            num = z[f"pairbin_num_{m}"][:, i]
            bs = [num[b].sum() / den[b, i].sum() for b in boots if den[b, i].sum() > 0]
            row[m] = {"purity": float(num.sum() / den[:, i].sum()), "ci95": np.percentile(bs, [2.5, 97.5]).tolist()}
        d = z["pairbin_num_H0"][:, i] - z["pairbin_num_NN"][:, i]
        row["H0_minus_NN"] = float(d.sum() / den[:, i].sum())
        out.append(row)
    return out


def stopped(z):
    res = {}
    for r in RHO_STOPS:
        d = {}
        for m in TREES + ("baseline",):
            for part in ("all", "active"):
                v = z[f"stop{r:g}_{m}_{part}"]
                d[f"{m}_{part}"] = float(np.nanmean(v))
        d["ln_omega_median"] = float(np.nanmedian(z[f"stop{r:g}_ln_omega"]))
        d["frac_sites_ge2"] = float(np.mean(z[f"stop{r:g}_frac_sites_ge2"]))
        res[f"{r:g}"] = d
    return res


def compare(a, b):
    """Reproducibility of the same realizations in two runs of the same code: realizations whose
    purity changes by more than 1e-9 (round-off of the free-fermion solution at strong disorder
    can reorder near-degenerate correlations), largest change and change of the mean."""
    res = {}
    for m in TREES:
        d = a[f"{m}_purity"] - b[f"{m}_purity"]
        res[m] = {"n_changed": int(np.sum(np.abs(d) > 1e-9)), "max_abs": float(np.nanmax(np.abs(d))),
                  "mean_shift": float(np.nanmean(d))}
    return res


def checks():
    res = {}
    for f in sorted(RUN.glob("fixed_logmean_*.npz")):
        a, b = np.load(f), np.load(C1 / f.name)
        n = a["seeds"].size
        assert np.array_equal(a["seeds"], b["seeds"][:n])
        res[f.stem] = compare(a, {k: b[k][:n] for k in (f"{m}_purity" for m in TREES)})
        res[f.stem]["cmin_changed"] = int(np.sum(a["cmin"] != b["cmin"][:n]))
    v2 = np.load(V2)
    for f in sorted(RUN.glob("box_L*.npz")):
        a = np.load(f)
        L = f.stem.split("L")[1]
        n = a["seeds"].size - 1
        res[f.stem] = compare({k: a[k][1:] for k in (f"{m}_purity" for m in TREES)},
                              {f"{m}_purity": v2[f"L{L}_{m}_purity"][:n] for m in TREES})
    return res


def plot(summary):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    key = "pair_strata"
    centers = lambda rows: [np.sqrt(max(r["rho_lo"], 1 / 64) * r["rho_hi"]) for r in rows]  # noqa: E731
    cols = {32: "C0", 64: "C1", 128: "C2", 256: "C3", 512: "C4"}
    ax = axs[0]
    for L in (32, 64, 128, 256, 512):
        rows = summary.get(f"box_L{L}", {}).get(key)
        if not rows:
            continue
        ok = [r for r in rows if r["n_real_with"] >= 20]
        x = centers(ok)
        ax.plot(x, [r["H0"]["purity"] for r in ok], "s-", c=cols[L], label=f"L={L} H0")
        ax.plot(x, [r["NN"]["purity"] for r in ok], "^:", c=cols[L])
        ax.plot(x, [r["average"]["purity"] for r in ok], "--", c=cols[L], lw=0.8)
    ax.set_title("box (paper ensemble)")
    ax = axs[1]
    for d, ls in ((0.25, "-"), (0.5, "--"), (1.0, "-."), (2.0, ":")):
        for L in (32, 256):
            rows = summary.get(f"fixed_logmean_d{d:g}_L{L}", {}).get(key)
            if not rows:
                continue
            ok = [r for r in rows if r["n_real_with"] >= 20]
            ax.plot(centers(ok), [r["H0"]["purity"] for r in ok], ls, marker="s", c=cols[L],
                    label=f"L={L} d={d:g}")
    ax.set_title("fixed_logmean, H0")
    for ax in axs[:2]:
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("SDRG joining stage rho (active clusters / L), late to the right")
        ax.set_ylabel("purity of the pairs joined at that stage")
        ax.legend(fontsize=7, ncol=2)
    ax = axs[2]
    for L in (32, 64, 128, 256, 512):
        s = summary.get(f"box_L{L}", {}).get("stopped")
        if not s:
            continue
        x = [float(k) for k in s]
        ax.plot(x, [s[k]["H0_all"] for k in s], "s-", c=cols[L], label=f"L={L} H0")
        ax.plot(x, [s[k]["NN_all"] for k in s], "^:", c=cols[L])
        ax.plot(x, [s[k]["baseline_all"] for k in s], "x", c=cols[L], alpha=0.5)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("stop at rho_stop")
    ax.set_ylabel("purity of the stopped partition")
    ax.set_title("box, stopped flow (x: baseline)")
    ax.legend(fontsize=7)
    fig.suptitle("SDRG pairs by joining stage, h=0, Gamma0=1 (solid H0, dotted NN, dashed average; bins with >= 20 realizations)")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"purity_by_rg_stage.{ext}", dpi=150)


def main():
    summary = {}
    for f in sorted(RUN.glob("*.npz")):
        z = np.load(f)
        summary[f.stem] = {"n_real": int(z["seeds"].size),
                           "global": {m: float(np.nanmean(z[f"{m}_purity"])) for m in TREES},
                           "qc": {"cmin_min": float(z["cmin"].min()), "cmax_excess_max": float(z["cmax_excess"].max()),
                                  "intra_tie_frac_max": float(z["intra_tie_frac"].max())},
                           "pair_strata": pair_strata(z), "strata": strata(z), "stopped": stopped(z)}
    summary["_checks_max_abs_diff_vs_previous_runs"] = checks()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(summary)
    for tag, s in summary.items():
        if tag.startswith("_"):
            continue
        line = " ".join(f"[{r['rho_hi']:.3g}] {r['H0']['purity']:.3f}/{r['NN']['purity']:.3f}"
                        f" w{r['pair_share']:.2f}" for r in s["pair_strata"])
        print(f"{tag:28s} glob {s['global']['H0']:.3f} | {line}")
    for tag, c in summary["_checks_max_abs_diff_vs_previous_runs"].items():
        print(f"check {tag:28s} " + " ".join(f"{m}: {c[m]['n_changed']} changed, max {c[m]['max_abs']:.1e}, "
                                            f"mean {c[m]['mean_shift']:+.1e}" for m in TREES))


if __name__ == "__main__":
    main()
