# %% [markdown]
# # R1 - Two kinds of boundary in the finite-size reentrant lobe (physics only)
#
# OBC, fixed_mean; L=16 (h=2.5, 97 reentrant seeds) and L=32 (h=2.3125, 137), and odd
# chains without boundary wall L=17 (h=2.5, 40) and L=33 (h=2.3125, 43). Interquartile
# bands are drawn for the even chains only, for readability.
# (a) staggered correlator at r=L/2-1 aligned at each seed's entry;
# (b) same, aligned at each seed's exit;
# (c) per-seed shift of the crossing when the correlator is measured at
#     another odd distance r, relative to r=L/2-1 (median, IQR band);
# (d) jump of -dE/dGamma0 across the crossing interval, in units of the
#     seed's typical per-step change (Hellmann-Feynman, <sigma^x_i> from MPS);
# (e) staggered |m_s| and uniform m_u magnetization aligned at the exit;
# (f) heterogeneity of the long-range pairs: fraction with C(i,i+r)>0 and
#     median of the strongest pair, aligned at the exit.

# %%
import json

import numpy as np
import matplotlib.pyplot as plt

import common as cm
from common import ps

ps.apply_style()
data = cm.load()
# odd chains (no boundary wall), prepared by prepare_data_oddL.py
data.update({L: dict(np.load(cm.DATA_DIR / f"reentrant_L{L}.npz", allow_pickle=False)) for L in (17, 33)})
data = dict(sorted(data.items()))
COLOR = {16: ps.SIZE_COLORS[16], 32: ps.SIZE_COLORS[32], 17: "#7b3294", 33: "#e66101"}
LS = {16: "--", 17: "--", 32: "-", 33: "-"}
BASENAME = "fig_reentrant_R1_two_boundaries"
summary = {}

fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4))

# (a)-(b) aligned correlator
for ax, key, title in ((axes[0, 0], "entry", "aligned at entry"), (axes[0, 1], "exit", "aligned at exit")):
    for L, d in data.items():
        g, step = d["Gamma0"], d["Gamma0"][1] - d["Gamma0"][0]
        cv = d["curves_r"][:, list(d["odd_r"]).index(int(d["R"]))]
        x, mean, samples = cm.aligned(cv, g, d[key], step)
        q1 = np.array([np.percentile(s, 25) if s.size else np.nan for s in samples])
        q3 = np.array([np.percentile(s, 75) if s.size else np.nan for s in samples])
        col = COLOR[L]
        if L % 2 == 0:
            ax.fill_between(x, q1, q3, color=col, alpha=0.18, lw=0)
        ax.plot(x, mean, LS[L], color=col, lw=1.8, label=fr"$L={L}$")
    ax.axvline(0, color="black", ls=":", lw=1.0)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set(xlabel=fr"$\Gamma_0-\Gamma_0^{{\rm {key}}}$", ylabel=r"$C_{\rm stag}(r=\lfloor L/2\rfloor-1)$", title=title,
           xlim=(-0.3, 0.3), ylim=(-0.6, 0.6))
    ax.legend(frameon=False, loc="upper left" if key == "entry" else "upper right")

# (c) scale dependence of the crossings
ax = axes[0, 2]
scale = {}
for L, d in data.items():
    g, R, odd_r = d["Gamma0"], int(d["R"]), list(d["odd_r"])
    ref = {k: d[k] for k in ("entry", "exit")}
    rs, med, lo, hi = {"entry": [], "exit": []}, {"entry": [], "exit": []}, {"entry": [], "exit": []}, {"entry": [], "exit": []}
    for j, r in enumerate(odd_r):
        if r == R:
            continue
        shifts = {"entry": [], "exit": []}
        for n in range(len(d["entry"])):
            (e, _), (x, _) = cm.entry_exit(g, d["curves_r"][n, j])
            shifts["entry"].append(abs(e - ref["entry"][n]))
            shifts["exit"].append(abs(x - ref["exit"][n]))
        for k in shifts:
            s = np.array(shifts[k])
            s = s[np.isfinite(s)]
            if s.size >= 8:
                rs[k].append(r / L); med[k].append(np.median(s))
                lo[k].append(np.percentile(s, 25)); hi[k].append(np.percentile(s, 75))
    col = COLOR[L]
    for k, mk, mfc in (("entry", "o", col), ("exit", "s", "white")):
        if L % 2 == 0:
            ax.fill_between(rs[k], lo[k], hi[k], color=col, alpha=0.12, lw=0)
        ax.plot(rs[k], med[k], LS[L], marker=mk, mfc=mfc, mec=col, color=col, ms=5,
                label=fr"$L={L}$, {k}")
    scale[f"L{L}"] = {k: {"r_over_L": rs[k], "median_shift": med[k]} for k in rs}
ax.set(yscale="log", xlabel=r"$r/L$", ylabel=r"$|\Gamma_0^\times(r)-\Gamma_0^\times(\lfloor L/2\rfloor-1)|$",
       title="dependence on the probing distance")
ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="upper left")
summary["scale_dependence"] = scale

# (d) energy-derivative jump
ax = axes[1, 0]
jumps = {}
for L, d in data.items():
    g = d["Gamma0"]
    mid = 0.5 * (g[1:] + g[:-1])
    dX = np.diff(d["minus_dE_dGamma0"], axis=1)
    typical = np.median(np.abs(dX), axis=1)
    col = COLOR[L]
    for k, ls in (("entry", "-"), ("exit", ":")):
        j = np.abs(np.array([dX[n, int(np.argmin(np.abs(mid - d[k][n])))] for n in range(len(d[k]))]) / typical)
        xs = np.sort(j)
        ax.step(xs, np.arange(1, xs.size + 1) / xs.size, where="post", color=col, ls=ls, lw=1.7,
                label=fr"$L={L}$, {k}")
        jumps[f"L{L}_{k}"] = {"median": float(np.median(j)), "fraction_gt_3": float(np.mean(j > 3))}
ax.axvline(1, color="0.5", lw=0.6)
ax.set(xscale="log", xlabel=r"$|\Delta(-\partial_{\Gamma_0}E)|$ at the crossing / typical step",
       ylabel="cumulative fraction of seeds", title="kink in the energy")
ax.legend(frameon=False, fontsize=7.6, loc="lower right")
summary["energy_derivative_jump"] = jumps

# (e) magnetization at the exit
ax = axes[1, 1]
for L, d in data.items():
    g, step = d["Gamma0"], d["Gamma0"][1] - d["Gamma0"][0]
    col = COLOR[L]
    for key, mk, lab in (("staggered_abs", "o", r"$|m_s|$"), ("uniform", "s", r"$m_u$")):
        x, mean, _ = cm.aligned(d[key], g, d["exit"], step)
        ax.plot(x, mean, LS[L], color=col, marker=mk, ms=3, mfc=col if key == "staggered_abs" else "white",
                label=fr"{lab}, $L={L}$")
ax.axvline(0, color="black", ls=":", lw=1.0)
ax.set(xlabel=r"$\Gamma_0-\Gamma_0^{\rm exit}$", ylabel="magnetization per site", title="staggered vs uniform",
       xlim=(-0.3, 0.3))
ax.legend(frameon=False, fontsize=7.6, ncol=2, loc="upper right")

# per seed: first crossing |m_s| = m_u after the entry (linear interpolation) versus the exit,
# and mean-field check C(r) ~ m_s^2 - m_u^2 within 0.1 of the exit
cross = {}
for L, d in data.items():
    g = d["Gamma0"]
    cv = d["curves_r"][:, list(d["odd_r"]).index(int(d["R"]))]
    ms, mu = d["staggered_abs"], d["uniform"]
    xs = []
    for n in range(d["exit"].size):
        f = ms[n] - mu[n]
        idx = [i for i in range(g.size - 1) if g[i] >= d["entry"][n] - 1e-9 and f[i] > 0 >= f[i + 1]]
        xs.append(g[idx[0]] + (g[idx[0] + 1] - g[idx[0]]) * f[idx[0]] / (f[idx[0]] - f[idx[0] + 1]) if idx else np.nan)
    xs = np.array(xs)
    near = np.abs(g[None, :] - d["exit"][:, None]) <= 0.1
    mf = ms ** 2 - mu ** 2
    cross[f"L{L}"] = {"found_fraction": float(np.isfinite(xs).mean()),
                      "median_abs_shift": float(np.nanmedian(np.abs(xs - d["exit"]))),
                      "median_signed_shift": float(np.nanmedian(xs - d["exit"])),
                      "grid_step": float(g[1] - g[0]),
                      "meanfield_median_abs_dev_near_exit": float(np.median(np.abs(cv - mf)[near])),
                      "meanfield_corr_near_exit": float(np.corrcoef(cv[near], mf[near])[0, 1])}
summary["exit_magnetization_crossing"] = cross

# (f) heterogeneity of long-range pairs after the exit
ax = axes[1, 2]
het = {}
for L, d in data.items():
    g, step = d["Gamma0"], d["Gamma0"][1] - d["Gamma0"][0]
    col = COLOR[L]
    x, frac, _ = cm.aligned(d["pair_fraction_positive"], g, d["exit"], step)
    _, cmax, _ = cm.aligned(d["pair_max"], g, d["exit"], step, statistic="median")
    ax.plot(x, frac, LS[L], color=col, lw=1.8, label=fr"pairs with $C>0$, $L={L}$")
    ax.plot(x, cmax, LS[L], color=col, lw=1.0, marker="^", ms=3, label=fr"strongest pair, $L={L}$")
    i0 = int(np.argmin(np.abs(x)))
    het[f"L{L}"] = {"fraction_positive_at_exit": float(frac[i0]), "strongest_pair_at_exit": float(cmax[i0])}
ax.axvline(0, color="black", ls=":", lw=1.0)
ax.axhline(0.5, color="0.6", lw=0.6)
ax.set(xlabel=r"$\Gamma_0-\Gamma_0^{\rm exit}$", ylabel=r"fraction / $C_{\rm stag}$", title=r"pairs at $r=\lfloor L/2\rfloor-1$",
       xlim=(-0.3, 0.3), ylim=(0, 1.0))
ax.legend(frameon=False, fontsize=7.2, loc="lower left")
summary["exit_heterogeneity"] = het

cm.lettered_titles(axes.ravel())
fig.tight_layout()
ps.save_figure(fig, cm.OUTPUT_DIR, BASENAME)
(cm.OUTPUT_DIR / f"{BASENAME}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary["energy_derivative_jump"], indent=1))
