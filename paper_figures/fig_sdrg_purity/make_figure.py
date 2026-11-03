# %% [markdown]
# # H0 hierarchy against the strong-disorder RG clusters (review v3)
#
# h = 0, Gamma0 = 1, OBC, C = Cstag raw, D = 1 - C, exact free fermions, 300 realizations per point.
# Half-column figure, two panels stacked vertically. Drawn at 6.4 x 7.0 in (as Fig. 3) so that, scaled to
# the column, fonts and markers match the full-width figures.
# (a) Dendrogram purity against 1/L for the box ensemble (fixed_mean, L = 32-512) for the H0
#     dendrogram, the nearest-neighbour dendrogram and an unrelated partition (former Fig. 3(c)).
# (b) Dendrogram purity against the disorder strength delta (power-law family, fixed mean
#     logarithms) for L = 32 and 256, H0 (solid) and nearest neighbour (dotted). Open symbols at
#     "box" are the box ensemble of panel (a) at the same sizes, placed at its std of ln J.
# 95 % bootstrap intervals over realizations.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent / "_common"))
import paper_style as ps  # noqa: E402

DATA_DIR = HERE / "data_raw"
OUTPUT_DIR = HERE / "output"
BASENAME = "fig_sdrg_purity"
N_BOOT = 2000
RNG = np.random.default_rng(20261018)
NN_COLOR = "#009E73"
DELTA_BOX = 0.2  # std of ln J for J uniform in [1, 2] (numbers.tex, DsDeltaBox)
DELTA_SIZES = {32: "#31688e", 256: "#e0a800"}

ps.apply_style()
ps.verify_manifest(DATA_DIR)
H = np.load(DATA_DIR / "sdrg_hierarchy_box.npz")
S = np.load(DATA_DIR / "sdrg_purity_delta.npz")
DELTAS = [float(d) for d in S["deltas"]]
summary = {}


def finite(v):
    v = np.asarray(v, float)
    return v[np.isfinite(v)]


def boot(v, stat=np.mean):
    b = [stat(v[RNG.integers(0, v.size, v.size)]) for _ in range(N_BOOT)]
    return float(stat(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.0))

# %% (a) H0 hierarchy vs SDRG clusters against 1/L (box ensemble)
ax = axes[0]
Lh = sorted(int(k.split("_")[0][1:]) for k in H.files if k.endswith("_H0"))
summary["sdrg_purity"] = {}
for key, lab, col, mk, ls in (("H0", r"$H_0$ hierarchy", ps.TOPOLOGY_COLOR, "s", "-"),
                              ("NN", "nearest-neighbour hierarchy", NN_COLOR, "^", "-"),
                              ("H0_baseline", "unrelated partition", "0.45", None, ":")):
    rows = [boot(finite(H[f"L{L}_{key}"][:, 1])) for L in Lh]
    m, lo, hi = (np.array(c) for c in zip(*rows))
    ax.errorbar(1.0 / np.array(Lh), m, yerr=[m - lo, hi - m], marker=mk, color=col, ls=ls, ms=6, lw=1.3,
                capsize=2, label=lab)
    summary["sdrg_purity"][key] = {str(L): r for L, r in zip(Lh, rows)}
ax.set(xlim=(0, 0.034), ylim=(0.3, 0.92), xlabel=r"$1/L$", ylabel="dendrogram purity")
ax.set_xticks([0] + [1.0 / L for L in (32, 64, 128)])
ax.set_xticklabels(["0"] + [f"1/{L}" for L in (32, 64, 128)])
ax.legend(frameon=False, fontsize=7.6, loc="lower right")

# %% (b) purity against the disorder strength delta
ax = axes[1]
summary["delta"] = {}
for L, col in DELTA_SIZES.items():
    for key, mk, ls in (("H0", "s", "-"), ("NN", "^", ":")):
        rows = [boot(finite(S[f"d{d:g}_L{L}_{key}_purity"])) for d in DELTAS]
        m, lo, hi = (np.array(c) for c in zip(*rows))
        ax.errorbar(DELTAS, m, yerr=[m - lo, hi - m], marker=mk, color=col, ls=ls, ms=6, lw=1.3, capsize=2)
        summary["delta"][f"L{L}_{key}"] = {f"{d:g}": r for d, r in zip(DELTAS, rows)}
        bkey = "H0" if key == "H0" else "NN"
        b = summary["sdrg_purity"][bkey][str(L)]
        ax.errorbar([DELTA_BOX], [b[0]], yerr=[[b[0] - b[1]], [b[2] - b[0]]], marker=mk, mfc="white",
                    color=col, ls="none", ms=6, capsize=2)
ax.set_xscale("log")
ax.set_xticks([DELTA_BOX, 0.25, 0.5, 1, 2])
ax.set_xticklabels(["box", "0.25", "0.5", "1", "2"])
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set(xlim=(0.17, 2.4), ylim=(0.55, 1.01), xlabel=r"disorder strength $\delta$",
       ylabel="dendrogram purity")
handles = [plt.Line2D([], [], color="0.2", marker="s", ls="-", ms=6, label=r"$H_0$ hierarchy"),
           plt.Line2D([], [], color="0.2", marker="^", ls=":", ms=6, label="nearest-neighbour hierarchy")]
handles += [plt.Line2D([], [], color=c, lw=3, label=fr"$L={L}$") for L, c in DELTA_SIZES.items()]
ax.legend(handles=handles, frameon=False, fontsize=7.6, loc="center right", bbox_to_anchor=(1.0, 0.36), ncol=2)

# %% Panel labels and output
ps.panel_labels(axes)
fig.tight_layout()
ps.save_figure(fig, OUTPUT_DIR, BASENAME)
(OUTPUT_DIR / f"{BASENAME}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
