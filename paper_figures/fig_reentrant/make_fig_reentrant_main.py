# %% [markdown]
# # Reentrant main figure - frontier points from physics and topology
#
# OBC, fixed_mean; L16 h=2.5 (97 reentrant seeds), L32 h=2.3125 (137); odd chains
# without boundary wall L17 h=2.5 (40 of 400), L33 h=2.3125 (43 of 400).
# C = Cstag (raw), topology on D = 1 - C. Physical boundaries: zero crossings
# of the half-chain staggered correlator C(r), r = L/2 - 1.
# Long-range (LR) landscape: H0 barcode of D restricted to the pairs with
# |d - r| <= w; level M_w = 1 - mean death. For w = 0 the shell is a forest
# and M_0 = C(r) exactly.
# (a) Frontier points: median over realizations of the per-seed estimate;
#     whiskers 95 % and thick bars 50 % seed-bootstrap intervals; horizontal
#     bands: physical 95 % intervals. Entry: global H0 landscape velocity
#     maximum and LR velocity maximum left of the apex of M_w. Exit: LR
#     velocity maximum right of the apex of M_w, second
#     wave of the H0 velocity (interior maxima only), maximum growth of the H1
#     cycle count N1 (tau = 0.006).
# (b) Wall parity of the averaged lobe (review v3, R1-4): disorder average of C(r), r = 7, over all
#     realizations at h = 2.5, rings (ED, 1000 seeds) and open chains (DMRG, 1000 / 400 seeds), L = 16, 17;
#     band 95 % seed bootstrap of the mean. Data: prepare_parity_average.py.

# %%
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import common as cm
from common import ps

ps.apply_style()
data = cm.load()
# odd chains (no boundary wall): L17 h=2.5, L33 h=2.3125, prepared by prepare_data_oddL.py
data.update({L: dict(np.load(cm.DATA_DIR / f"reentrant_L{L}.npz", allow_pickle=False)) for L in (17, 33)})
GROUP_X = {32: 0.0, 33: 0.2, 16: 0.52, 17: 0.72}      # x position of each size group in panel (a)
BASENAME = "fig_reentrant_main"
RNG = np.random.default_rng(20261014)
N_BOOT = 5000
COL = {"phys": "0.1", "H0": ps.TOPOLOGY_COLOR, "H1": "#762a83", 0: "#009E73", 1: "#CC79A7", 2: "#56B4E9"}
MRK = {"phys": "o", "H0": "s", "H1": "h", 0: "D", 1: "^", 2: "v"}
LABEL = {"phys": r"physical: zero of $C(\lfloor L/2\rfloor-1)$ ($\equiv$ zero of $M_0$)", "H0": r"$H_0$ landscape velocity (all pairs)", "H1": r"$H_1$: first new cycle after the entry ($\tau=0.04$)",
         0: r"long-range landscape, $w=0$", 1: r"long-range landscape, $w=1$ ($L=16,17$)",
         2: r"long-range landscape, $w=2$ ($L=32,33$)"}
WIDE = {16: 1, 17: 1, 32: 2, 33: 2}
OFFSET = {"phys": 0.0, "H0": 0.03, 0: 0.06, "wide": 0.09, "H1": 0.12}
summary = {}


def frontier(values):
    """Median of per-seed estimates and seed-bootstrap 95 % / 50 % intervals."""
    values = np.asarray(values, float)
    draws = RNG.integers(0, values.size, (N_BOOT, values.size))
    boot = np.nanmedian(values[draws], axis=1)
    return {"median": float(np.nanmedian(values)), "N": int(np.isfinite(values).sum()), "of": int(values.size),
            "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
            "ci50": [float(v) for v in np.percentile(boot, [25, 75])]}


H1_COMMON = np.load(cm.DATA_DIR / "reentrant_H1_common_tau.npz")   # tau shared with the clean frontier
H1_TAU = float(H1_COMMON["tau"])


def h1_growth_peak(d):
    """Per seed: first interval after the physical entry where N1 (lifetimes > H1_TAU) increases.

    Same rule and threshold as the first H1 cycle of the clean frontier (Sec. V); the entry is
    used only to start the search, since cycles can already exist in the disordered phase below it.
    """
    g, mid = d["Gamma0"], d["Gamma0_mid"]
    counts = H1_COMMON[f"L{int(d['L'])}_H1_count"]
    out = []
    for n1, e in zip(counts, d["entry"]):
        ok = (mid > e) & (np.diff(n1) >= 1)
        out.append(mid[np.flatnonzero(ok)[0]] if ok.any() else np.nan)
    return np.array(out)


def draw_point(ax, x, r, key, filled):
    c = COL[key]
    ax.plot([x, x], r["ci95"], color=c, lw=1.1, zorder=3)
    ax.plot([x, x], r["ci50"], color=c, lw=4.5, alpha=0.35, solid_capstyle="butt", zorder=3)
    ax.plot(x, r["median"], MRK[key], color=c, ms=7.5, mfc=c if filled else "white", mew=1.5, zorder=4)


# review v3: one column, panels stacked (the former panel (b), one realization, was removed)
fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(6.4, 9.0), gridspec_kw={"height_ratios": (1.3, 1.0)})

# %% (a) frontier points
for L, d in sorted(data.items()):
    h = float(d["h"])
    x_base = GROUP_X[L]
    wide = WIDE[L]
    pts = {"entry": {"phys": frontier(d["entry"]), "H0": frontier(cm.topological_entry(d)),
                     0: frontier(cm.split_velocity(d, 0)[0]), wide: frontier(cm.split_velocity(d, wide)[0])},
           "exit": {"phys": frontier(d["exit"]), "H0": frontier(cm.second_wave(d)[0]), "H1": frontier(h1_growth_peak(d)), 0: frontier(cm.split_velocity(d, 0)[1]), wide: frontier(cm.split_velocity(d, wide)[1])}}
    x0, x1 = x_base - 0.018, x_base + OFFSET["H1"] + 0.018
    for branch in ("entry", "exit"):
        lo, hi = pts[branch]["phys"]["ci95"]
        ax_a.fill_between([x0, x1], lo, hi, color="0.88", lw=0, zorder=0)
        for key, r in pts[branch].items():
            off = OFFSET["wide"] if key == wide else OFFSET[key]
            draw_point(ax_a, x_base + off, r, key, filled=branch == "entry")
    ax_a.text(x_base + OFFSET["H1"] / 2, 0.585, fr"$L={L}$", ha="center", fontsize=10)
    lv = cm.level_crossings(d, wide)[1]
    summary[f"L{L}"] = {"h": h, "seeds": int(d["entry"].size),
                        "entry": {str(k): v for k, v in pts["entry"].items()},
                        "exit": {str(k): v for k, v in pts["exit"].items()},
                        f"exit_zero_of_level_w{wide}_fraction": float(np.isfinite(lv).mean()),
                        "exit_LR_velocity_w0_right_of_apex": frontier(cm.split_velocity(d, 0)[1]),
                        "exit_H0_second_wave": frontier(cm.second_wave(d)[0]),
                        "per_seed_entry_MAE": {k: float(np.nanmean(np.abs(v - d["entry"]))) for k, v in
                                               (("H0", cm.topological_entry(d)), ("w0", cm.split_velocity(d, 0)[0]),
                                                (f"w{wide}", cm.split_velocity(d, wide)[0]))},
                        "per_seed_exit_MAE": {"H0_second_wave": float(np.nanmean(np.abs(cm.second_wave(d)[0] - d["exit"]))),
                                              "H1_growth": float(np.nanmean(np.abs(h1_growth_peak(d) - d["exit"]))),
                                              "LR_w0": float(np.nanmean(np.abs(cm.split_velocity(d, 0)[1] - d["exit"]))),
                                              f"LR_w{wide}": float(np.nanmean(np.abs(cm.split_velocity(d, wide)[1] - d["exit"]))),
                                              "constant": float(np.mean(np.abs(np.median(d["exit"]) - d["exit"])))},
                        "identity_max_abs_M0_minus_C_r": float(np.abs(d["LR_level_w0"] - d["curves_r"][:, list(d["odd_r"]).index(int(d["R"]))]).max())}
ax_a.set_xlim(-0.04, 0.88)
ax_a.set_ylim(0.0, 0.61)
pair_centres = {h: 0.5 * (GROUP_X[a] + GROUP_X[b]) + OFFSET["H1"] / 2 for h, (a, b) in ((2.3125, (32, 33)), (2.5, (16, 17)))}
ax_a.set_xticks(list(pair_centres.values()))
ax_a.set_xticklabels([f"{h:g}" for h in pair_centres])
for xg in (GROUP_X[33] - 0.04, GROUP_X[17] - 0.04):
    ax_a.axvline(xg, color="0.75", lw=0.6, ls=":")
ax_a.set_xlabel(r"longitudinal field $h$ (even $L$ left, odd $L$ right)")
ax_a.set_ylabel(r"frontier point $\Gamma_0$")
handles = [Line2D([], [], marker=MRK[k], color=COL[k], ls="", ms=7, label=LABEL[k]) for k in ("phys", "H0", 0, 1, 2, "H1")]
handles += [Line2D([], [], marker="o", color="0.4", ls="", ms=7, label="entry"),
            Line2D([], [], marker="o", color="0.4", mfc="white", ls="", ms=7, label="exit")]
ax_a.legend(handles=handles, fontsize=7.6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, columnspacing=0.8, handletextpad=0.3)
ax_a.set_title("reentrant frontier points")

# %% (b) wall parity of the averaged lobe
par = np.load(cm.DATA_DIR / "reentrant_parity_average.npz")
g = par["Gamma0"]
PARITY_COLOR = {"pairs": "#0072B2", "forced": "#E69F00"}
summary["parity_average"] = {}
for bc, L, walls, ls in (("pbc", 16, "pairs", "-"), ("obc", 17, "pairs", "--"),
                         ("pbc", 17, "forced", "-"), ("obc", 16, "forced", "--")):
    c = par[f"{bc}_L{L}_C"]
    draws = RNG.integers(0, c.shape[0], (N_BOOT, c.shape[0]))
    lo, hi = np.percentile(c[draws].mean(1), [2.5, 97.5], axis=0)
    mean = c.mean(0)
    name = {"pbc": "ring", "obc": "open"}[bc]
    ax_b.fill_between(g, lo, hi, color=PARITY_COLOR[walls], alpha=0.18, lw=0)
    ax_b.plot(g, mean, color=PARITY_COLOR[walls], ls=ls, lw=1.6, label=fr"{name}, $L={L}$ ({c.shape[0]})")
    summary["parity_average"][f"{bc}_L{L}"] = {"N": int(c.shape[0]), "max": float(mean.max()), "argmax": float(g[mean.argmax()]),
                                               "at_0": float(mean[0]), "zeros": cm.crossings(g, mean)}
ax_b.axhline(0, color="0.3", lw=0.7)
ax_b.set_xlim(0, float(g[-1]))
ax_b.set_xlabel(r"$\Gamma_0$")
ax_b.set_ylabel(r"$\overline{C}(r)$, $r=7$")
handles, labels = ax_b.get_legend_handles_labels()
ax_b.legend(handles, labels, fontsize=7.6, frameon=False, loc="lower left", ncol=2, columnspacing=1.0,
            title=r"walls in pairs (blue), forced wall (orange)", title_fontsize=7.6)
ax_b.set_title(r"disorder average, all realizations, $h=2.5$")

cm.lettered_titles([ax_a, ax_b])
fig.tight_layout(h_pad=1.2)
ps.save_figure(fig, cm.OUTPUT_DIR, BASENAME)
(cm.OUTPUT_DIR / f"{BASENAME}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=1))
