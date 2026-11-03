# %% [markdown]
# # R3 - H1 marks where the antiferromagnetic order yields locally
#
# (a) cumulative distribution of the coupling J on the middle bond of the
#     four-site H1 motifs (square gap > tau = 0.006) at the largest field,
#     against all bulk bonds;
# (b) fraction of reentrant seeds with at least one H1 cycle (frozen H1 cycle
#     counts, tau = 0.006) versus Gamma0 - Gamma0_exit;
# (c) one L = 32 seed: square gap g_i(Gamma0) along the chain (positive values
#     only), with the couplings J_i on the right and entry/exit marked.

# %%
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import common as cm
from common import ps

ps.apply_style()
data = cm.load()
BASENAME = "fig_reentrant_R3_H1_local"
TAU = 0.006
EXAMPLE_L = 32
summary = {}

fig = plt.figure(figsize=(13.2, 4.1))
grid = fig.add_gridspec(1, 4, width_ratios=(1.0, 1.0, 1.0, 0.32), wspace=0.30)
ax_a, ax_b, ax_c = (fig.add_subplot(grid[0, i]) for i in range(3))
ax_j = fig.add_subplot(grid[0, 3], sharey=ax_c)

# (a) motif location
for L, d in data.items():
    col = ps.SIZE_COLORS[L]
    active = d["square_gap"][:, -1, :] > TAU                       # (n, L-3) at the largest field
    J_mid = d["J"][:, 1:L - 2]                                     # middle bond of motif s is bond s+1
    J_motif, J_all = J_mid[active], J_mid.ravel()
    for vals, ls, lab in ((J_all, ":", "all bulk bonds"), (J_motif, cm.LS[L], r"$H_1$ motifs")):
        xs = np.sort(vals)
        ax_a.step(xs, np.arange(1, xs.size + 1) / xs.size, where="post", color=col, ls=ls, lw=1.8,
                  label=fr"{lab}, $L={L}$")
    summary[f"L{L}_motif_J"] = {"n_motifs": int(J_motif.size), "median_motif": float(np.median(J_motif)),
                                "median_all": float(np.median(J_all)), "mann_whitney_p": float(mannwhitneyu(J_motif, J_all).pvalue)}
ax_a.set(xlabel=r"coupling $J$ on the middle bond", ylabel="cumulative fraction", title=r"$H_1$ sits on weak bonds")
ax_a.legend(frameon=False, fontsize=7.4, loc="upper left")

# (b) activation across the exit
for L, d in data.items():
    g, step = d["Gamma0"], d["Gamma0"][1] - d["Gamma0"][0]
    x, frac, _ = cm.aligned((d["H1_count"] > 0).astype(float), g, d["exit"], step)
    ax_b.plot(x, frac, cm.LS[L], color=ps.SIZE_COLORS[L], lw=1.8, marker="o", ms=3, label=fr"$L={L}$")
    summary[f"L{L}_fraction_with_H1"] = {f"{v:+.2f}": float(frac[int(np.argmin(np.abs(x - v)))]) for v in (-0.1, 0.0, 0.05)}
ax_b.axvline(0, color="black", ls=":", lw=1.0)
ax_b.set(xlabel=r"$\Gamma_0-\Gamma_0^{\rm exit}$", ylabel=r"fraction of seeds with $H_1$", ylim=(0, 1.02), xlim=(-0.3, 0.3),
         title="activation across the exit")
ax_b.legend(frameon=False, loc="upper left")

# (c) spatial example: the L=32 seed with most motif-bearing sites after the exit
d = data[EXAMPLE_L]
g = d["Gamma0"]
counts = [(d["square_gap"][n][g > d["exit"][n]] > TAU).any(axis=0).sum() for n in range(len(d["exit"]))]
n = int(np.argmax(counts))
gap = np.maximum(d["square_gap"][n], 0.0)                          # (G, L-3)
L = EXAMPLE_L
im = ax_c.imshow(gap.T, origin="lower", aspect="auto", cmap="Purples",
                 extent=[g[0] - (g[1] - g[0]) / 2, g[-1] + (g[1] - g[0]) / 2, 1.5, L - 2.5], vmin=0)
ax_c.axvline(d["entry"][n], color="#0072B2", ls="--", lw=1.2, label="entry")
ax_c.axvline(d["exit"][n], color="#D55E00", ls="--", lw=1.2, label="exit")
ax_c.set(xlabel=r"$\Gamma_0$", ylabel="middle bond of the motif", title=f"seed {int(d['seeds'][n])}, $L={L}$")
ax_c.legend(frameon=False, fontsize=7.6, loc="upper left")
cb = fig.colorbar(im, ax=ax_j, fraction=0.6, pad=0.15)
cb.set_label("square gap $g_i$", fontsize=8)
ax_j.barh(np.arange(L - 1), d["J"][n], height=0.8, color="0.6")   # bond k = middle bond index
ax_j.set(xlabel=r"$J_i$", xlim=(0.8, 2.2))
ax_j.tick_params(labelleft=False)
summary["example"] = {"L": L, "seed": int(d["seeds"][n]), "entry": float(d["entry"][n]), "exit": float(d["exit"][n])}

cm.lettered_titles([ax_a, ax_b, ax_c])
ps.save_figure(fig, cm.OUTPUT_DIR, BASENAME)
(cm.OUTPUT_DIR / f"{BASENAME}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=1))
