# %% [markdown]
# # Random chain at h = 0 (V1): pseudocritical points
#
# OBC, fixed_mean, C = Cstag raw, topology on D = 1 - C, exact free fermions.
# 1000 realizations for L = 16-128, 500 for L = 256.
# Review v3: half-column figure, (a) above (b), drawn at 6.4 x 7.0 in. The former panel (c), H0 hierarchy vs SDRG clusters, is now panel (a) of
# ../fig_sdrg_purity. data_raw/random_h0_sdrg_hierarchy.npz stays (hashed) but is not plotted here.
# (a) Distribution of the topological pseudocritical points (maximum of xi_lambda of each
#     realization), one ridge per size, vertical line at the mean; dashed line Gamma0_c = 1.
# (b) Width sigma(L) of the distributions of the topological and physical pseudocritical points
#     against 1/L (log-log), 95 % bootstrap intervals; dotted guide L^{-1/2} (IDFP).

# %%
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent / "_common"))
import paper_style as ps  # noqa: E402

DATA_DIR = HERE / "data_raw"
OUTPUT_DIR = HERE / "output"
BASENAME = "fig_random_h0"
N_BOOT = 2000
RNG = np.random.default_rng(20261018)

ps.apply_style()
ps.verify_manifest(DATA_DIR)
P = np.load(DATA_DIR / "random_h0_peaks.npz")
SIZES = [int(L) for L in P["sizes"]]
summary = {}


def finite(v):
    v = np.asarray(v, float)
    return v[np.isfinite(v)]


def boot(v, stat):
    b = [stat(v[RNG.integers(0, v.size, v.size)]) for _ in range(N_BOOT)]
    return float(stat(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.0))  # review v3: half column, panels stacked

# %% (a) ridgeline of the topological pseudocritical points
ax = axes[0]
grid = np.linspace(0.75, 1.12, 400)
for k, L in enumerate(SIZES):
    p = finite(P[f"L{L}_xi"])
    dens = gaussian_kde(p)(grid)
    top = k + 0.85 * dens / dens.max()
    ax.fill_between(grid, k, top, color=ps.TOPOLOGY_COLOR, alpha=0.35, lw=0)
    ax.plot(grid, top, color=ps.TOPOLOGY_COLOR, lw=1.2)
    ax.plot([p.mean()] * 2, [k, k + 0.85], color="0.15", lw=1.0)
    ax.text(0.755, k + 0.15, fr"$L={L}$", fontsize=8.5, va="bottom")
    summary[f"L{L}"] = {"xi_mean": float(p.mean()), "n": int(p.size)}
ax.axvline(1.0, color="0.2", ls="--", lw=1.0)
ax.set(xlim=(0.75, 1.12), yticks=[], xlabel=r"$\Gamma^\star_{0,\alpha}$ (maximum of $\xi_\lambda$ per realization)")

# %% (b) widths against 1/L
ax = axes[1]
for key, lab, col, mk in (("xi", r"$\xi_\lambda$ ($H_0$)", ps.TOPOLOGY_COLOR, "s"),
                          ("phys", r"$-\partial_{\Gamma_0}m^2_{s,\mathcal{B}}$", ps.PHYSICAL_COLOR, "o")):
    rows = [boot(finite(P[f"L{L}_{key}"]), lambda v: v.std(ddof=1)) for L in SIZES]
    w, lo, hi = (np.array(c) for c in zip(*rows))
    ax.errorbar(1.0 / np.array(SIZES), w, yerr=[w - lo, hi - w], marker=mk, color=col, ms=6, lw=1.3, capsize=2, label=lab)
    for L, (v, a, b) in zip(SIZES, rows):
        summary[f"L{L}"][f"width_{key}"] = [v, a, b]
ref = summary[f"L{SIZES[-1]}"]["width_xi"][0]
Ls = np.array([SIZES[0], SIZES[-1]], float)
ax.plot(1.0 / Ls, ref * (Ls / SIZES[-1]) ** -0.5, ":", color="0.3", lw=1.2, label=r"$L^{-1/2}$ (IDFP)")
ax.set(xscale="log", yscale="log", xlabel=r"$1/L$", ylabel=r"width $\sigma(\Gamma^\star_{0,\alpha})$")
ax.set_xticks(1.0 / np.array(SIZES))
ax.set_xticklabels([f"1/{L}" for L in SIZES])
ax.minorticks_off()
ax.set_yticks([0.015, 0.02, 0.03, 0.04, 0.05])
ax.set_yticklabels(["0.015", "0.02", "0.03", "0.04", "0.05"])
ax.legend(frameon=False, fontsize=7.6, loc="upper left")

# %% Panel labels and output
ps.panel_labels(axes)
fig.tight_layout()
ps.save_figure(fig, OUTPUT_DIR, BASENAME)
(OUTPUT_DIR / f"{BASENAME}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# %%
fig
