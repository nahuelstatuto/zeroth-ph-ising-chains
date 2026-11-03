"""Exploratory mock-ups (review v3) of a half-column cluster figure.

Panel (a) is the former Fig. 3(c), purity against 1/L. Panel (b) has three
candidates: P1 purity against delta (C1), P2 purity by SDRG merge stage (B2,
box ensemble), P3 purity by merge stage for several delta at L=256.
Output only to results/review_v3/figure_proposals_clusters (not paper_figures).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper_figures/_common"))
import paper_style as ps  # noqa: E402

OUT = ROOT / "results/review_v3/figure_proposals_clusters"
H = np.load(ROOT / "paper_figures/fig_random_h0/data_raw/random_h0_sdrg_hierarchy.npz")
DS = ROOT / "results/review_v3/sdrg-disorder-strength-v1"
RS = json.load(open(ROOT / "results/review_v3/sdrg-rg-scale-v1/analysis/summary.json"))
NN_COLOR = "#009E73"
RNG = np.random.default_rng(20261102)
FIGSIZE = (3.4, 5.4)
L_COLORS = {32: "#31688e", 64: "#35b779", 128: "#90d743", 256: "#e0a800", 512: "#b5179e"}
D_COLORS = {0.25: "#440154", 0.5: "#3b528b", 1.0: "#21918c", 2.0: "#5ec962"}
ps.apply_style()


def boot(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    b = [v[RNG.integers(0, v.size, v.size)].mean() for _ in range(1000)]
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def panel_a(ax):
    Lh = sorted(int(k.split("_")[0][1:]) for k in H.files if k.endswith("_H0"))
    for key, lab, col, mk, ls in (("H0", r"$H_0$", ps.TOPOLOGY_COLOR, "s", "-"),
                                  ("NN", "nearest neighbour", NN_COLOR, "^", "-"),
                                  ("H0_baseline", "unrelated", "0.45", None, ":")):
        m, lo, hi = (np.array(c) for c in zip(*[boot(H[f"L{L}_{key}"][:, 1]) for L in Lh]))
        ax.errorbar(1 / np.array(Lh), m, yerr=[m - lo, hi - m], marker=mk, color=col, ls=ls,
                    ms=4, lw=1.2, capsize=1.5, label=lab)
    ax.set(xlim=(0, 0.034), ylim=(0.3, 0.92), xlabel=r"$1/L$", ylabel="purity")
    ax.set_xticks([0] + [1 / L for L in (32, 64, 128)])
    ax.set_xticklabels(["0"] + [f"1/{L}" for L in (32, 64, 128)])
    ax.legend(frameon=False, fontsize=7, loc="lower right")


def panel_delta(ax):
    deltas = [0.25, 0.5, 0.75, 1, 1.5, 2]
    for L, ms in ((32, 3.5), (256, 4)):
        for key, mk, ls in (("H0", "s", "-"), ("NN", "^", ":")):
            m, lo, hi = (np.array(c) for c in zip(*[boot(np.load(DS / f"fixed_logmean_d{d:g}_L{L}.npz")[f"{key}_purity"]) for d in deltas]))
            ax.errorbar(deltas, m, yerr=[m - lo, hi - m], marker=mk, ls=ls, color=L_COLORS[L], ms=ms,
                        lw=1.2, capsize=1.5, label=f"$L={L}$" if key == "H0" else None)
        box = RS[f"box_L{L}"]["global"]
        ax.plot([0.2, 0.2], [box["H0"], box["NN"]], ls="none", marker="o", mfc="none", color=L_COLORS[L], ms=4)
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.25, 0.5, 1, 2])
    ax.set_xticklabels(["box", "", "0.5", "1", "2"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set(xlabel=r"disorder strength $\delta$", ylabel="purity", ylim=(0.55, 1.01))
    ax.plot([], [], "ks-", ms=3, label=r"$H_0$")
    ax.plot([], [], "k^:", ms=3, label="nearest neighbour")
    ax.legend(frameon=False, fontsize=7, loc="lower right", ncol=2)


def panel_stage_box(ax):
    for L in (32, 128, 512):
        st = sorted(RS[f"box_L{L}"]["pair_strata"], key=lambda s: -s["rho_hi"])
        x = np.array([np.sqrt(s["rho_lo"] * s["rho_hi"]) if s["rho_lo"] > 0 else s["rho_hi"] / 1.41 for s in st])
        for key, mk, ls in (("H0", "s", "-"), ("NN", "^", ":")):
            m = np.array([s[key]["purity"] for s in st])
            lo = np.array([s[key]["ci95"][0] for s in st])
            hi = np.array([s[key]["ci95"][1] for s in st])
            ax.errorbar(x, m, yerr=[m - lo, hi - m], marker=mk, ls=ls, color=L_COLORS[L], ms=3.5, lw=1.2,
                        capsize=1.2, label=f"$L={L}$" if key == "H0" else None)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set(xlabel=r"merge stage $\rho$ (late $\rightarrow$)", ylabel="purity of the pairs", ylim=(0.3, 1.01))
    ax.plot([], [], "ks-", ms=3, label=r"$H_0$")
    ax.plot([], [], "k^:", ms=3, label="nearest neighbour")
    ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)


def panel_stage_delta(ax):
    for d in (0.25, 0.5, 1.0, 2.0):
        st = sorted(RS[f"fixed_logmean_d{d:g}_L256"]["pair_strata"], key=lambda s: -s["rho_hi"])
        x = np.array([np.sqrt(s["rho_lo"] * s["rho_hi"]) for s in st])
        ax.plot(x, [s["H0"]["purity"] for s in st], "s-", color=D_COLORS[d], ms=3.5, lw=1.2, label=fr"$\delta={d:g}$")
    st = sorted(RS["box_L256"]["pair_strata"], key=lambda s: -s["rho_hi"])
    x = np.array([np.sqrt(s["rho_lo"] * s["rho_hi"]) for s in st])
    ax.plot(x, [s["H0"]["purity"] for s in st], "s-", color=ps.TOPOLOGY_COLOR, ms=3.5, lw=1.2, label=r"box, $H_0$")
    ax.plot(x, [s["NN"]["purity"] for s in st], "^:", color=NN_COLOR, ms=3.5, lw=1.2, label="box, n. neighbour")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set(xlabel=r"merge stage $\rho$ (late $\rightarrow$), $L=256$", ylabel="purity of the pairs", ylim=(0.3, 1.01))
    ax.legend(frameon=False, fontsize=6.6, loc="lower left", ncol=2)


for name, panel_b in (("P1_delta", panel_delta), ("P2_stage_box", panel_stage_box), ("P3_stage_delta", panel_stage_delta)):
    fig, (a, b) = plt.subplots(2, 1, figsize=FIGSIZE)
    panel_a(a)
    panel_b(b)
    for ax, t in ((a, "(a)"), (b, "(b)")):
        ax.set_title(t, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
print("ok")
