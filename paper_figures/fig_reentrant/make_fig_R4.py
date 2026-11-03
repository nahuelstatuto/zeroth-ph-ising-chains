# %% [markdown]
# # R4 - What topology says about the exit
#
# Aligned at each realization's exit (OBC; L16 h=2.5, 97 seeds; L32 h=2.3125, 137).
# (a) The antiferromagnetic skeleton survives the exit: largest connected
#     fraction of the sign skeleton {odd |i-j|, C_ij > 0} (mean, 5-95 % band)
#     and the H0 backbone level (mean C on the edges of the H0 forest), with
#     the half-chain correlator as reference (it crosses zero by definition).
# (b) Local breaks arrive after the exit: fraction of half-chain windows
#     [i, i+L/2-1] containing a four-site H1 motif (lifetime > 0.006) and
#     fraction of realizations with at least one H1 cycle.

# %%
import json

import numpy as np
import matplotlib.pyplot as plt

import common as cm
from common import ps

ps.apply_style()
data = cm.load()
BASENAME = "fig_reentrant_R4_exit_topology"
summary = {}

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
ax_a, ax_b = axes

for L, d in data.items():
    g, step = d["Gamma0"], d["Gamma0"][1] - d["Gamma0"][0]
    col = ps.SIZE_COLORS[L]
    # (a)
    x, perc, samples = cm.aligned(d["skeleton_percolation"], g, d["exit"], step)
    p5 = np.array([np.percentile(s, 5) if s.size else np.nan for s in samples])
    p95 = np.array([np.percentile(s, 95) if s.size else np.nan for s in samples])
    ax_a.fill_between(x, p5, p95, color=col, alpha=0.15, lw=0)
    ax_a.plot(x, perc, cm.LS[L], color=col, lw=2.0, label=fr"sign-skeleton percolation, $L={L}$")
    _, backbone, _ = cm.aligned(d["H0_backbone_level"], g, d["exit"], step)
    ax_a.plot(x, backbone, cm.LS[L], color=col, lw=1.2, marker="s", ms=3, label=fr"$H_0$ backbone level, $L={L}$")
    cv = d["curves_r"][:, list(d["odd_r"]).index(int(d["R"]))]
    _, ref, _ = cm.aligned(cv, g, d["exit"], step)
    ax_a.plot(x, ref, cm.LS[L], color="0.55", lw=1.0, label=fr"$C_{{\rm stag}}(L/2-1)$, $L={L}$")
    i0 = int(np.argmin(np.abs(x)))
    summary[f"L{L}"] = {"skeleton_percolation_at_exit": float(perc[i0]),
                        "skeleton_percolation_min_over_seeds_all_fields_after_entry": float(np.nanmin(
                            np.where(g[None, :] > d["entry"][:, None], d["skeleton_percolation"], np.nan))),
                        "backbone_level_at_exit": float(backbone[i0])}
    # (b)
    x, cov, _ = cm.aligned(d["motif_coverage"], g, d["exit"], step)
    ax_b.plot(x, cov, cm.LS[L], color=col, lw=2.0, label=fr"windows with an $H_1$ motif, $L={L}$")
    _, frac, _ = cm.aligned((d["H1_count"] > 0).astype(float), g, d["exit"], step)
    ax_b.plot(x, frac, cm.LS[L], color=col, lw=1.0, marker="o", ms=3, label=fr"realizations with $H_1$, $L={L}$")
    cross = x[np.flatnonzero((cov[:-1] < 0.5) & (cov[1:] >= 0.5))]
    summary[f"L{L}"]["motif_coverage_half_crossing_offset"] = float(cross[0]) if cross.size else None

for ax in axes:
    ax.axvline(0, color="black", ls=":", lw=1.0)
    ax.axhline(0.5, color="0.7", lw=0.6)
ax_a.axhline(0, color="0.7", lw=0.6)
ax_a.set(xlabel=r"$\Gamma_0-\Gamma_0^{\rm exit}$", ylabel="fraction / correlation", xlim=(-0.3, 0.3), ylim=(-0.35, 1.05),
         title="the antiferromagnetic skeleton survives")
ax_a.legend(frameon=False, fontsize=7.0, loc="center left", bbox_to_anchor=(0.0, 0.47), ncol=2)
ax_b.set(xlabel=r"$\Gamma_0-\Gamma_0^{\rm exit}$", ylabel="fraction", xlim=(-0.3, 0.3), ylim=(0, 1.02),
         title="local breaks arrive after the exit")
ax_b.legend(frameon=False, fontsize=7.4, loc="upper left")

cm.lettered_titles(axes)
fig.tight_layout()
ps.save_figure(fig, cm.OUTPUT_DIR, BASENAME)
(cm.OUTPUT_DIR / f"{BASENAME}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=1))
