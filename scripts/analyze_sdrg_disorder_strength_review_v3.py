"""Analysis of run_sdrg_disorder_strength_review_v3.py (referee 1, objection 2).

QC rule, fixed before writing the text: a block (ensemble, delta, L) is valid when every
realization satisfies the Griffiths bounds within 1e-3 (min C >= -1e-3, max C <= 1 + 1e-3) and
no intra-cluster SDRG pair falls below the elimination cutoff. Invalid blocks are kept in the
files and in the figure (open symbols) but not quoted.
Run:  python scripts/analyze_sdrg_disorder_strength_review_v3.py
Output: results/review_v3/sdrg-disorder-strength-v1/analysis/{qc_table.json, macros.json,
        purity_vs_delta.png/.pdf}
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "review_v3" / "sdrg-disorder-strength-v1"
OUT = RUN / "analysis"
SIZES = (32, 64, 128, 256)
DELTAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
TOL = 1e-3


def valid(s):
    q = s["qc"]
    return q["cmin_min"] >= -TOL and q["cmax_excess_max"] <= TOL and q["intra_tie_frac_max"] == 0.0


def main():
    S = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    qc = {k: {"valid": valid(v), **v["qc"]} for k, v in S.items()}
    (OUT / "qc_table.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, ens in zip(axes, ("fixed_logmean", "raw")):
        for i, L in enumerate(SIZES):
            c = f"C{i}"
            for m, ls, mk in (("H0", "-", "s"), ("average", "--", None), ("NN", ":", "^")):
                xs = [d for d in DELTAS if f"{ens}_d{d:g}_L{L}" in S]
                ys = [S[f"{ens}_d{d:g}_L{L}"][m]["purity"]["mean"] for d in xs]
                ok = [valid(S[f"{ens}_d{d:g}_L{L}"]) for d in xs]
                ax.plot([x for x, o in zip(xs, ok) if o], [y for y, o in zip(ys, ok) if o], ls=ls, color=c,
                        marker=mk, ms=4, label=f"L={L} {m}" if m != "average" else None)
                if mk:
                    ax.plot([x for x, o in zip(xs, ok) if not o], [y for y, o in zip(ys, ok) if not o],
                            ls="none", marker=mk, ms=5, mfc="none", color=c)
        ax.axvline(0.2, color="grey", lw=0.8, ls="-.")
        ax.set_xscale("log")
        ax.set_xticks((0.2,) + DELTAS)
        ax.set_xticklabels(["box"] + [f"{d:g}" for d in DELTAS], fontsize=7)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(r"$\delta$")
        ax.set_title(ens)
    axes[0].set_ylabel("dendrogram purity")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("SDRG clusters vs hierarchies, power-law disorder, h=0, Gamma0=1 (open: QC invalid; dash-dot: box ensemble)",
                 fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"purity_vs_delta.{ext}", dpi=200)

    f = lambda k, m="H0": S[k][m]["purity"]["mean"]  # noqa: E731
    mac = {
        "DsDeltaBox": 0.20,
        "DsPurTopLmaxDsmall": f("fixed_logmean_d0.25_L256"),
        "DsPurTopLmaxDone": f("fixed_logmean_d1_L256"),
        "DsPurTopLmaxDtwo": f("fixed_logmean_d2_L256"),
        "DsPurTopLminDtwo": f("fixed_logmean_d2_L32"),
        "DsPurNnLmaxLo": min(f(f"fixed_logmean_d{d:g}_L256", "NN") for d in DELTAS[:6]),
        "DsPurNnLmaxHi": max(f(f"fixed_logmean_d{d:g}_L256", "NN") for d in DELTAS[:6]),
        "DsPurAvgDiffMax": max(abs(S[f"fixed_logmean_d{d:g}_L{L}"]["H0_minus_average_purity"]["mean"])
                               for d in DELTAS[:6] for L in SIZES),
        "DsPurRawMaxDiff": max(abs(f(f"fixed_logmean_d{d:g}_L{L}") - f(f"raw_d{d:g}_L{L}"))
                               for d in DELTAS[:6] for L in SIZES),
        "DsNonNnLmaxDsmall": S["fixed_logmean_d0.25_L256"]["sdrg_stats"]["nonnn_merges"],
        "DsNonNnLmaxDtwo": S["fixed_logmean_d2_L256"]["sdrg_stats"]["nonnn_merges"],
        "DsInvalid": sorted(k for k, v in qc.items() if not v["valid"]),
    }
    (OUT / "macros.json").write_text(json.dumps(mac, indent=2), encoding="utf-8")
    print(json.dumps(mac, indent=1))


if __name__ == "__main__":
    main()
