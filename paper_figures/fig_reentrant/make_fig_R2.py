# %% [markdown]
# # R2 - What the H0 landscape velocity sees at the two boundaries
#
# (a) H0 landscape velocity (frozen arrays), log of the per-seed normalized
#     value, aligned at each seed's entry; mean over seeds, 95% bootstrap band;
# (b) same aligned at each seed's exit, using only fields beyond entry + 2
#     grid steps; grey bands: the same curve with exits permuted among seeds
#     (95% of 500 permutations);
# (c) localization error versus the intrinsic width of the physical crossing
#     (lobe amplitude / |slope|): topological estimate (entry = frozen global
#     maximum; exit = second wave, interior maxima only) and the physical
#     ambiguity of the crossing when r -> r +/- 2. Lines: binned medians.

# %%
import json

import numpy as np
import matplotlib.pyplot as plt

import common as cm
from common import ps

ps.apply_style()
data = cm.load()
BASENAME = "fig_reentrant_R2_topology"
RNG = np.random.default_rng(20261014)
N_BOOT, N_PERM = 1000, 500
summary = {}


def per_seed_binned(values, x, ref, step, mask=None):
    bins = np.arange(-cm.ALIGN_RANGE - step / 2, cm.ALIGN_RANGE + step, step)
    centres = 0.5 * (bins[1:] + bins[:-1])
    out = np.full((len(ref), centres.size), np.nan)
    for n in range(len(ref)):
        dx = x - ref[n]
        ok = np.ones_like(x, bool) if mask is None else mask[n]
        for b in range(centres.size):
            sel = ok & (dx >= bins[b]) & (dx < bins[b + 1])
            if sel.any():
                out[n, b] = values[n][sel].mean()
    return centres, out


def boot_band(mat):
    means = np.array([np.nanmean(mat[RNG.integers(0, len(mat), len(mat))], axis=0) for _ in range(N_BOOT)])
    return np.nanpercentile(means, [2.5, 97.5], axis=0)


fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), gridspec_kw={"width_ratios": (1, 1, 1.05)})
for L, d in data.items():
    col = ps.SIZE_COLORS[L]
    vlog = cm.log_normalized_velocity(d)
    mid = d["Gamma0_mid"]
    step = d["Gamma0"][1] - d["Gamma0"][0]
    # (a) entry
    x, mat = per_seed_binned(vlog, mid, d["entry"], step)
    lo, hi = boot_band(mat)
    axes[0].fill_between(x, lo, hi, color=col, alpha=0.25, lw=0)
    axes[0].plot(x, np.nanmean(mat, axis=0), cm.LS[L], color=col, lw=1.8, label=fr"$L={L}$")
    # (b) exit, beyond the entry
    after = mid[None, :] > d["entry"][:, None] + 2 * step
    x, mat = per_seed_binned(vlog, mid, d["exit"], step, after)
    lo, hi = boot_band(mat)
    null = np.array([np.nanmean(per_seed_binned(vlog, mid, d["exit"][RNG.permutation(len(d["exit"]))], step, after)[1], axis=0)
                     for _ in range(N_PERM)])
    nlo, nhi = np.nanpercentile(null, [2.5, 97.5], axis=0)
    axes[1].fill_between(x, nlo, nhi, color="0.6", alpha=0.25, lw=0)
    axes[1].fill_between(x, lo, hi, color=col, alpha=0.25, lw=0)
    axes[1].plot(x, np.nanmean(mat, axis=0), cm.LS[L], color=col, lw=1.8, label=fr"$L={L}$")
    i0 = int(np.argmin(np.abs(x)))
    summary[f"L{L}_exit_aligned"] = {"value_at_exit": float(np.nanmean(mat, axis=0)[i0]),
                                     "null_95_at_exit": [float(nlo[i0]), float(nhi[i0])]}

for ax, key in ((axes[0], "entry"), (axes[1], "exit")):
    ax.axvline(0, color="black", ls=":", lw=1.0)
    ax.set(xlabel=fr"$\Gamma_0-\Gamma_0^{{\rm {key}}}$", ylabel=r"$\ln[\xi_\lambda/{\rm median}(\xi_\lambda)]$",
           title=f"aligned at {key}", xlim=(-0.3, 0.3))
    ax.legend(frameon=False, loc="upper right")
axes[1].text(0.03, 0.04, "grey: exits permuted among seeds", transform=axes[1].transAxes, fontsize=7.6, color="0.35")

# (c) sharpness law
ax = axes[2]
pooled = {"entry": ([], [], []), "exit": ([], [], [])}
for L, d in data.items():
    geo = cm.crossing_geometry(d)
    top_e = cm.topological_entry(d)
    sw, status = cm.second_wave(d)
    width_e, width_x = geo["amp"] / geo["slope_entry"], geo["amp"] / geo["slope_exit"]
    err_e = np.abs(top_e - d["entry"])
    err_x = np.where(status == "interior", np.abs(sw - d["exit"]), np.nan)
    for k, w, e, a in (("entry", width_e, err_e, geo["amb_entry"]), ("exit", width_x, err_x, geo["amb_exit"])):
        pooled[k][0].extend(w); pooled[k][1].extend(e); pooled[k][2].extend(a)
        ax.scatter(w, e, s=9, marker="o" if L == 32 else "D", color=("#0072B2" if k == "entry" else "#D55E00"),
                   alpha=0.35, lw=0)
w_all = np.concatenate([pooled[k][0] for k in pooled])
e_all = np.concatenate([pooled[k][1] for k in pooled])
a_all = np.concatenate([pooled[k][2] for k in pooled])
bins = np.geomspace(3e-3, 1.0, 11)
centres = np.sqrt(bins[1:] * bins[:-1])
for vals, style, lab in ((e_all, "-", "topology (binned median)"), (a_all, "--", r"physics, $r\to r\pm2$ (binned median)")):
    med = []
    for lo_, hi_ in zip(bins[:-1], bins[1:]):
        s = vals[(w_all >= lo_) & (w_all < hi_) & np.isfinite(vals)]
        med.append(np.median(s) if s.size >= 6 else np.nan)
    ax.plot(centres, med, style, color="black", lw=1.8, marker="o", ms=4, label=lab)
ax.plot([3e-3, 1], [3e-3, 1], ":", color="0.5", lw=1.0, label="error = width")
ax.scatter([], [], s=12, color="#0072B2", label="entries")
ax.scatter([], [], s=12, color="#D55E00", label="exits")
ax.set(xscale="log", yscale="log", xlim=(3e-3, 1), ylim=(5e-4, 0.5), xlabel="intrinsic crossing width",
       ylabel="localization error", title="precision follows sharpness")
ax.legend(frameon=False, fontsize=7.4, loc="upper left")
summary["sharpness_law_spearman_width_vs_topological_error"] = float(
    __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(w_all[np.isfinite(e_all)], e_all[np.isfinite(e_all)]).statistic)

cm.lettered_titles(axes)
fig.tight_layout()
ps.save_figure(fig, cm.OUTPUT_DIR, BASENAME)
(cm.OUTPUT_DIR / f"{BASENAME}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=1))
