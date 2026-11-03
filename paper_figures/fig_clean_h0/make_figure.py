# %% [markdown]
# # Clean chain at h = 0 (V1, exact free fermions): physical response, xi_lambda, finite-size fits
#
# OBC, J = 1, g = Gamma/J, C = Cstag raw, topology on D = 1 - C. L = 16-512, grid step 0.005.
# (a) -d m^2_{s,B}/dg; (b) xi_lambda(g, L) from Eq. (6) of the paper (closed form of the landscape
# velocity for H0, checked against the landscape integral in v0); (c) g*(L) against 1/L with
# unweighted fits g* = g_c + a/L + b/L^2 on L >= 32. Both estimators use the same peak rule
# (discrete argmax + local three-point parabola). Further fits (a/L on L >= 64, free exponent)
# are written to the summary.
#
# Review v3 (B5): panel (c) and all fits use the maxima refined off the simulation grid (`data_raw/clean_h0_refined_peaks.npz`); the summary also reports the systematics over ansatzes and fit windows.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent / "_common"))
import paper_style as ps  # noqa: E402

DATA_DIR = HERE / "data_raw"
OUTPUT_DIR = HERE / "output"
BASENAME = "fig_clean_h0"
FIT_MIN = 32

ps.apply_style()
ps.verify_manifest(DATA_DIR)
z = np.load(DATA_DIR / "clean_h0_exact.npz")
SIZES = [int(L) for L in z["sizes"]]

# %% Observables
curves = {}
for L in SIZES:
    g, deaths, m2 = z[f"L{L}_g"], z[f"L{L}_deaths"], z[f"L{L}_m2_bulk"]
    dd = np.gradient(deaths, g, axis=0, edge_order=2)
    xi = np.sqrt((deaths * dd ** 2).sum(axis=1) / (2 * (L - 1)))        # Eq. (6)
    resp = -np.gradient(m2, g, edge_order=2)
    curves[L] = {"g": g, "response": resp, "xi": xi,
                 "peak_response": ps.quadratic_peak(g, resp), "peak_xi": ps.quadratic_peak(g, xi)}

sizes = np.array(SIZES, float)
# Review v3 (B5): panel (c) and the fits use the maxima refined off the 0.005 grid
# (data_raw/clean_h0_refined_peaks.npz). The grid maxima of panels (a) and (b) are biased by up to
# 6e-4 for the largest chains and are kept only for display and in "fits_grid".
R = np.load(DATA_DIR / "clean_h0_refined_peaks.npz")
assert [int(L) for L in R["sizes"]] == SIZES
g_phys = np.asarray(R["physical"], float)
g_top = np.asarray(R["xi"], float)
g_phys_grid = np.array([curves[L]["peak_response"]["x"] for L in SIZES])
g_top_grid = np.array([curves[L]["peak_xi"]["x"] for L in SIZES])
use = sizes >= FIT_MIN


def fit(model, p0, x, y):
    popt, pcov = curve_fit(model, x, y, p0=p0, maxfev=20000)
    return [float(v) for v in popt], [float(v) for v in np.sqrt(np.diag(pcov))]


quad = lambda L, gc, a, b: gc + a / L + b / L ** 2      # noqa: E731
lin = lambda L, gc, a: gc + a / L                       # noqa: E731
free = lambda L, gc, a, inv_nu: gc + a * L ** (-inv_nu)  # noqa: E731
log_lead = lambda L, gc, a, c: gc + (a + c * np.log(L)) / L      # noqa: E731
log_sub = lambda L, gc, a, c: gc + a / L + c * np.log(L) / L ** 2  # noqa: E731


def fit_set(y):
    out = {"quadratic_Lmin32": fit(quad, [1, -1, 0], sizes[use], y[use]),
           "linear_Lmin64": fit(lin, [1, -1], sizes[sizes >= 64], y[sizes >= 64]),
           "free_exponent_Lmin64": fit(free, [1, -1, 1], sizes[sizes >= 64], y[sizes >= 64])}
    # systematics (R1-7, R2-MC2): three-parameter ansatzes, L >= 16, 32, 64 (at least four sizes)
    band = {}
    for tag, model in (("quadratic", quad), ("log_subleading", log_sub), ("log_leading", log_lead), ("free_exponent", free)):
        for lmin in (16, 32, 64):
            m = sizes >= lmin
            band[f"{tag}_Lmin{lmin}"] = fit(model, [1, -1, 1 if tag == "free_exponent" else 0], sizes[m], y[m])
    out["systematics"] = band
    gcs = [v[0][0] for v in band.values()]
    out["gc_range"] = [min(gcs), max(gcs)]
    out["inverse_nu_free_range"] = [min(band[f"free_exponent_Lmin{l}"][0][2] for l in (16, 32, 64)),
                                    max(band[f"free_exponent_Lmin{l}"][0][2] for l in (16, 32, 64))]
    out["local_inverse_nu_gc1"] = (np.log((1 - y[:-1]) / (1 - y[1:])) / np.log(sizes[1:] / sizes[:-1])).tolist()
    return out


fits = {"physical": fit_set(g_phys), "landscape_velocity": fit_set(g_top)}
fits_grid = {"physical": fit_set(g_phys_grid), "landscape_velocity": fit_set(g_top_grid)}
print(json.dumps({k: v["quadratic_Lmin32"] for k, v in fits.items()}, indent=1))

# %% Figure
fig, axes = plt.subplots(1, 3, figsize=ps.FIGSIZE_THREE_PANEL, gridspec_kw={"width_ratios": ps.WIDTH_RATIOS_THREE_PANEL})
for ax, key, peak_key, marker, ylabel in (
    (axes[0], "response", "peak_response", "o", r"$-\partial_g\, m^{2}_{s,\mathcal{B}}$"),
    (axes[1], "xi", "peak_xi", "s", r"$\xi_\lambda(g,L)/\sqrt{L-1}$"),
):
    for L in SIZES:
        c = curves[L]
        ax.plot(c["g"], c[key], color=ps.SIZE_COLORS[L], lw=1.4, label=fr"$L={L}$")
        ax.scatter(c[peak_key]["x"], c[peak_key]["value"], marker=marker, s=50,
                   facecolor=ps.SIZE_COLORS[L], edgecolor="black", linewidth=0.8, zorder=5)
    ax.axvline(1.0, color="black", ls="--", lw=1.1, label=r"$g_c=1$")
    ax.set(xlabel=r"$g=\Gamma/J$", ylabel=ylabel, xlim=(0.6, 1.4))
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, loc="upper right", fontsize=7.8)
    ax.grid(alpha=0.2)

ax = axes[2]
x_line = np.linspace(1e-4, 1.0 / FIT_MIN * 1.05, 200)
for values, name, color, marker, ls, label in (
    (g_phys, "physical", ps.PHYSICAL_COLOR, "o", "-", r"max $-\partial_g m^2_{s,\mathcal{B}}$"),
    (g_top, "landscape_velocity", ps.TOPOLOGY_COLOR, "s", "--", r"max $\xi_\lambda$"),
):
    (gc, a, b), (egc, _, _) = fits[name]["quadratic_Lmin32"]
    ax.plot(1.0 / sizes[use], values[use], marker, color=color, ms=6, label=label)
    ax.plot(1.0 / sizes[~use], values[~use], marker, mfc="white", mec=color, ms=6)
    ax.plot(x_line, quad(1.0 / x_line, gc, a, b), ls, color=color, lw=1.4,
            label=fr"fit Eq. (9), $L\geq{FIT_MIN}$")
ax.axhline(1.0, color="black", ls="--", lw=1.1, label=r"exact $g_c=1$")
ax.set(xlabel=r"$1/L$", ylabel=r"$g^\star(L)$", xlim=(-0.002, 0.066))
ax.legend(frameon=False, loc="lower left", fontsize=7.6)
ax.grid(alpha=0.2)

ps.panel_labels(axes)
fig.tight_layout()
ps.save_figure(fig, OUTPUT_DIR, BASENAME)

summary = {
    "figure": BASENAME, "method": "exact free fermions (Lieb-Schultz-Mattis), clean OBC chain at h=0",
    "xi_definition": "closed form Eq. (6) of the paper on the simulated grid",
    "peak_rule": "panels (a), (b): discrete argmax + local three-point parabola on the 0.005 grid; panel (c) and fits: refined maxima (data_raw/clean_h0_refined_peaks.npz)",
    "refined_peaks": {"physical": g_phys.tolist(), "landscape_velocity": g_top.tolist()},
    "derivative": "numpy.gradient edge_order=2",
    "sizes": {str(L): {"grid_step": float(np.min(np.diff(curves[L]["g"]))),
                       "physical_peak": curves[L]["peak_response"], "landscape_velocity_peak": curves[L]["peak_xi"]}
              for L in SIZES},
    "fits": fits,
    "fits_grid": fits_grid,
    "xi_peak_height_vs_lnL": dict(zip(["slope", "intercept"],
                                      [float(v) for v in np.polyfit(np.log(sizes), [curves[L]["peak_xi"]["value"] for L in SIZES], 1)])),
}
(OUTPUT_DIR / f"{BASENAME}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
