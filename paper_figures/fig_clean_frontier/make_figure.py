# %% [markdown]
# # Clean frontier (h > 0, OBC): H0 landscape velocity and the first H1 cycle
#
# (a) (h, Gamma) plane: digitized thermodynamic boundary (Lajko et al. 2020,
#     Fig. 1); maxima of the H0 landscape velocity xi_lambda (squares) for
#     even L = 16, 32, 64; onset of the first persistent H1 cycle (circles)
#     for odd L = 31, 63.
# (b) L = 63, h = 1.7, full-Gamma scan: physical response, xi_lambda and the
#     number of H1 cycles N1, normalized to their maxima.
# (c) Same cut: end-end and end-centre staggered correlations. The cycle is
#     the chain closed through its ends; it appears when the ends stay
#     correlated with each other while the end-centre correlation decays.
#
# H0 rules shared with the other figures: highest chi per (L, h); derivatives
# with numpy.gradient on the simulated grid; global maximum refined by a local
# three-point parabola. QC rule: a point is not a maximum candidate if it or a
# neighbour has branch disagreement or bond saturation; flags are kept.
#
# H1 rule (amendment v2 of documentation/PLAN_CLEAN_FRONTIER_ODDL_CLOSURE_20261015.md,
# fixed 2026-10-15 12:30 UTC on L31 h=1.1-1.4 and L63 h=1.3, 1.7; the other
# 12 odd-L cuts were a blind test): all H1 intervals of the Vietoris-Rips
# filtration of D, N1 = # lifetimes > tau = 0.04; estimate = midpoint of the
# first QC-valid Gamma interval where N1 increases. Odd L only: with even L
# and h > J the open chain carries a forced domain wall.
#
# Cuts with h >= 2(1 - 1/L) are controls for even L; L = 64, h = 1.95 is not
# converged and not drawn.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent / "_common"))
import paper_style as ps  # noqa: E402

DATA_DIR = HERE / "data_raw"
OUTPUT_DIR = HERE / "output"
BASENAME = "fig_clean_frontier"
REF_H = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.9, 1.95, 2.0])
REF_G = np.array([1.0, 0.992, 0.970, 0.931, 0.878, 0.802, 0.712, 0.599, 0.453, 0.278, 0.154, 0.077, 0.0])
SIZES = (16, 32, 64)
ODD_SIZES = (31, 63)
H1_TAU = 0.04
H1_COLOR = {63: "#762a83", 31: "#c2a5cf"}
H1_TRAIN = {(31, 1.1), (31, 1.2), (31, 1.3), (31, 1.4), (63, 1.3), (63, 1.7)}
EXAMPLE = (63, 1.7)
EXAMPLE_GAMMA_MIN = 0.1
QC_CUTS = {(64, 1.95)}
H_OFFSET = {16: -0.022, 32: 0.0, 64: 0.022, 31: 0.035, 63: 0.035}   # visual offset in h only
TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

ps.apply_style()
ps.verify_manifest(DATA_DIR)

# %% Estimators per cut


def h0_velocity(g, deaths, L):
    t = np.linspace(0.0, max(1.0, 1.02 * deaths.max()), 2001)
    layers = np.sort(np.maximum(0.0, np.minimum(t[None, None, :], deaths[:, :, None] - t[None, None, :])), axis=1)
    d_layers = np.gradient(layers, g, axis=0, edge_order=2)
    return np.sqrt(TRAPEZOID((d_layers ** 2).sum(axis=1), t, axis=1) / (L - 1))


def masked_peak(g, y, allowed):
    i = int(np.argmax(np.where(allowed, y, -np.inf)))
    edge = i in (0, len(g) - 1) or not (allowed[max(i - 1, 0)] and allowed[min(i + 1, len(g) - 1)])
    if edge:
        return {"x": float(g[i]), "value": float(y[i]), "edge_limited": True}
    p = ps.quadratic_peak(g[i - 1:i + 2], y[i - 1:i + 2])
    return {"x": p["x"], "value": p["value"], "edge_limited": False}


def h1_first_rise(g, counts, flat, flag, tau):
    """N1(tau) over all intervals; midpoint of the first QC-valid interval where N1 rises."""
    intervals = np.split(flat.reshape(-1, 2), np.cumsum(counts)[:-1])
    n1 = np.array([int(np.sum(iv[:, 1] - iv[:, 0] > tau)) if len(iv) else 0 for iv in intervals])
    rise = (np.diff(n1) >= 1) & ~(flag[:-1] | flag[1:])
    if not rise.any():
        return n1, None
    i = int(np.flatnonzero(rise)[0])
    return n1, {"x": float(0.5 * (g[i] + g[i + 1])), "edge_limited": i == 0}


def load_cuts(filename, sizes, odd):
    z = np.load(DATA_DIR / filename)
    best = {}
    for entry in json.loads(str(z["index_json"])):
        key = (entry["L"], round(entry["h"], 4))
        if entry["L"] in sizes and (key not in best or entry["chi"] > best[key]["chi"]):
            best[key] = entry
    out = {}
    for (L, h), e in sorted(best.items()):
        p = e["prefix"]
        g = np.asarray(z[f"{p}_Gamma"], float)
        if g.size < 4 or (L, h) in QC_CUTS or (not odd and h >= 2 * (1 - 1 / L) - 1e-12):
            continue
        flag = np.asarray(z[f"{p}_branch_disagreement"], bool) | (np.asarray(z[f"{p}_max_bond"]) >= e["chi"])
        allowed = ~(flag | np.r_[False, flag[:-1]] | np.r_[flag[1:], False])
        response = -np.gradient(np.asarray(z[f"{p}_bulk_ms2"], float), g, edge_order=2)
        xi0 = h0_velocity(g, np.asarray(z[f"{p}_H0_deaths"], float), L)
        cut = {"chi": e["chi"], "g": g, "flag": flag, "allowed": allowed, "response": response, "xi0": xi0,
               "full_gamma": e["includes_full_gamma_scan"],
               "reference": float(np.interp(h, REF_H, REF_G)),
               "reference_interpolated": not np.any(np.isclose(REF_H, h)),
               "peak_physical": masked_peak(g, response, allowed),
               "peak_H0": masked_peak(g, xi0, allowed), "peak_H1": None}
        if odd:
            cut["n1"], cut["peak_H1"] = h1_first_rise(g, np.asarray(z[f"{p}_H1_counts"], int),
                                                      np.asarray(z[f"{p}_H1_intervals"], float), flag, H1_TAU)
            cut["C_end_end"] = np.asarray(z[f"{p}_C_end_end"], float)
            cut["C_end_centre"] = np.asarray(z[f"{p}_C_end_centre"], float)
            flat = np.asarray(z[f"{p}_H1_intervals"], float).reshape(-1, 2)
            cut["max_H1_lifetime"] = float((flat[:, 1] - flat[:, 0]).max()) if len(flat) else 0.0
            for name in ("mz_end_first", "mz_end_last", "mz_centre"):
                cut[name] = np.asarray(z[f"{p}_{name}"], float)
        out[(L, h)] = cut
    return out


cuts = load_cuts("clean_frontier_compact.npz", SIZES, odd=False)
all_odd = load_cuts("clean_frontier_oddL_compact.npz", ODD_SIZES, odd=True)
GRID_H = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9}   # odd-L closure grid (train/test)
odd_cuts = {k: c for k, c in all_odd.items() if round(k[1], 4) in GRID_H}
extra_cuts = {k: c for k, c in all_odd.items() if round(k[1], 4) not in GRID_H}
cmp_cuts = {k: c for k, c in extra_cuts.items() if k[1] > 1.9}     # near the multicritical point (L63)
low_cuts = {k: c for k, c in extra_cuts.items() if k[1] < 1.0}     # low-field controls (L31)
CMP_SLOPE = 1.526492   # Gamma_c ~ 1.5265 (2J - h) near the multicritical point (Lajko et al. 2020)
FIT_HMIN = 1.9          # linear fit Gamma* = a (h_c - h) on the L = 63 cuts with h >= FIT_HMIN


def linear_fit(hs, gs):
    """Unweighted fit Gamma* = a (h_c - h); returns a, h_c and their standard errors."""
    hs, gs = np.asarray(hs, float), np.asarray(gs, float)
    (m, c), cov = np.polyfit(hs, gs, 1, cov=True)
    a, hc = -m, -c / m
    jac = np.array([[-1.0, 0.0], [c / m ** 2, -1.0 / m]])     # d(a, hc)/d(m, c)
    err = np.sqrt(np.diag(jac @ cov @ jac.T))
    resid = gs - np.polyval([m, c], hs)
    return {"a": float(a), "a_err": float(err[0]), "h_c": float(hc), "h_c_err": float(err[1]),
            "rmse": float(np.sqrt(np.mean(resid ** 2))), "h": hs.tolist(), "Gamma": gs.tolist()}


def mae(table, L, kind, keys=None):
    errs = [abs(c[f"peak_{kind}"]["x"] - c["reference"]) for (LL, h), c in table.items()
            if LL == L and c[f"peak_{kind}"] is not None and (keys is None or (LL, h) in keys)]
    return (float(np.mean(errs)), len(errs)) if errs else (None, 0)


metrics = {f"L{L}": {kind: mae(cuts, L, kind) for kind in ("physical", "H0")} for L in SIZES}
test_keys = {k for k in odd_cuts if k not in H1_TRAIN}
for L in ODD_SIZES:
    metrics[f"L{L}"] = {"physical": mae(odd_cuts, L, "physical"), "H0": mae(odd_cuts, L, "H0"),
                        "H1": mae(odd_cuts, L, "H1"), "H1_test": mae(odd_cuts, L, "H1", test_keys),
                        "H1_train": mae(odd_cuts, L, "H1", H1_TRAIN)}
print(json.dumps(metrics, indent=1))

# %% Figure
fig, axes = plt.subplots(1, 3, figsize=ps.FIGSIZE_THREE_PANEL, gridspec_kw={"width_ratios": (1.25, 1.0, 0.85)})

# (a) phase plane
ax = axes[0]
ax.fill_between(REF_H, 0, REF_G, color="0.93", zorder=0)
ax.plot(REF_H, REF_G, color="black", lw=1.4, zorder=2)
ax.plot(REF_H[1:-1], REF_G[1:-1], "o", color="black", ms=2.6, zorder=2)
ax.text(0.95, 0.55, "AFM order", fontsize=9, color="0.35")
ax.text(0.80, 1.08, "paramagnet", fontsize=9, color="0.35")
for (L, h), c in cuts.items():
    # V1 (referee 2): full-Gamma scans large with a thick black edge, windows small and translucent
    pk, color, full = c["peak_H0"], ps.SIZE_COLORS[L], c["full_gamma"]
    ax.scatter(h + H_OFFSET[L], pk["x"], marker="s", s=40 if full else 18,
               facecolor="white" if pk["edge_limited"] else color,
               edgecolor="black" if full else color, alpha=1.0 if full else 0.75,
               linewidth=1.5 if full else 0.8, zorder=4 if full else 3.5)
for L in ODD_SIZES:
    pts = sorted((h + H_OFFSET[L], c["peak_H1"]["x"]) for (LL, h), c in odd_cuts.items()
                 if LL == L and c["peak_H1"] is not None)
    filled = L == max(ODD_SIZES)
    ax.plot(*np.array(pts).T, ls="--" if filled else ":", lw=1.1, color=H1_COLOR[L], zorder=4.5)
    ax.scatter(*np.array(pts).T, marker="o", s=60 if filled else 50, color="white", zorder=4.6)   # halo
    ax.scatter(*np.array(pts).T, marker="o", s=34 if filled else 28,
               facecolor=H1_COLOR[L] if filled else "white", edgecolor="black" if filled else H1_COLOR[L],
               linewidth=0.6 if filled else 1.3, zorder=5)
ax.axvline(EXAMPLE[1], color="0.5", ls=":", lw=1.0, zorder=1)
ax.text(EXAMPLE[1] + 0.02, 0.02, "(b,c)", fontsize=8, color="0.4")

# inset: approach to the multicritical point (L = 63, narrow windows)
ins = ax.inset_axes([0.65, 0.63, 0.33, 0.33])
hh = np.linspace(1.88, 2.0, 50)
ins.plot(hh, np.interp(hh, REF_H, REF_G), color="black", lw=1.2)
ins.plot(hh, CMP_SLOPE * (2 - hh), color="0.45", ls="--", lw=0.9)
near = {k: c for k, c in all_odd.items() if k[0] == 63 and k[1] >= 1.9}
xs = sorted(near)
ins.plot([k[1] for k in xs], [near[k]["peak_H0"]["x"] for k in xs], "D", ms=4, color=ps.TOPOLOGY_COLOR, mec="black", mew=0.4)
ins.plot([k[1] for k in xs], [near[k]["peak_H1"]["x"] for k in xs], "o", ms=4, color=H1_COLOR[63], mec="black", mew=0.4)
fits = {kind: linear_fit([k[1] for k in xs if k[1] >= FIT_HMIN - 1e-9],
                         [near[k][f"peak_{kind}"]["x"] for k in xs if k[1] >= FIT_HMIN - 1e-9])
        for kind in ("H0", "physical", "H1")}
for kind, color in (("H0", ps.TOPOLOGY_COLOR), ("H1", H1_COLOR[63])):
    f = fits[kind]
    ins.plot(hh, f["a"] * (f["h_c"] - hh), color=color, lw=0.9, alpha=0.8)
ins.set(xlim=(1.88, 2.0), ylim=(0, 0.17))
ins.tick_params(labelsize=6)
ins.set_title(r"$L=63$ near $(2J,0)$", fontsize=6.5, pad=2)
ins.grid(alpha=0.2)

blank = plt.Line2D([], [], ls="", marker="")
handles = [plt.Line2D([], [], color="black", lw=1.4), blank,
           tuple(plt.Line2D([], [], ls="", marker="s", ms=6.5, color=ps.SIZE_COLORS[L], mec="black", mew=1.5)
                 for L in SIZES),
           tuple(plt.Line2D([], [], ls="", marker="s", ms=4.2, color=ps.SIZE_COLORS[L], alpha=0.75)
                 for L in SIZES),
           plt.Line2D([], [], ls="", marker="s", ms=4.2, mfc="white", mec="0.4"), blank,
           plt.Line2D([], [], ls="--", lw=1.1, marker="o", color=H1_COLOR[63], mec="black", mew=0.6),
           plt.Line2D([], [], ls=":", lw=1.1, marker="o", mfc="white", color=H1_COLOR[31], mew=1.3)]
labels = ["thermodynamic boundary", r"$H_0$, even $L$: max $\xi_\lambda$",
          r"$L=16,\ 32,\ 64$, full-$\Gamma$ scan", r"$L=16,\ 32,\ 64$, window",
          "maximum at window edge",
          fr"$H_1$, odd $L$: first cycle, $\tau={H1_TAU:g}$", r"$L=63$", r"$L=31$"]
ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
          frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=7.2)
ax.set(xlabel=r"$h/J$", ylabel=r"$\Gamma/J$", xlim=(0, 2.05), ylim=(0, 1.2))
ax.grid(alpha=0.2)

# (b) full-Gamma cut at odd L: physical response, xi_lambda, N1
ax = axes[1]
c = odd_cuts[EXAMPLE]
show = c["g"] >= EXAMPLE_GAMMA_MIN
if c["flag"][show].any():
    raise ValueError("Flagged states inside the displayed range of the example cut")
g = c["g"][show]
for key, color, marker, label in (("response", ps.PHYSICAL_COLOR, "o", r"$-\partial_\Gamma m^2_{s,\mathcal{B}}$"),
                                  ("xi0", ps.TOPOLOGY_COLOR, "s", r"$\xi_\lambda$ ($H_0$)")):
    y = c[key][show]
    ax.plot(g, y / y.max(), marker=marker, ms=2.2, lw=1.3, color=color, label=label)
n1 = c["n1"][show]
ax.step(g, n1 / n1.max(), where="post", color=H1_COLOR[63], lw=1.4, label=fr"$N_1$ ($H_1$, $\tau={H1_TAU:g}$)")
ax.scatter(c["peak_physical"]["x"], 1.0, marker="o", s=55, color=ps.PHYSICAL_COLOR, edgecolor="black", zorder=5)
ax.scatter(c["peak_H0"]["x"], 1.0, marker="s", s=55, color=ps.TOPOLOGY_COLOR, edgecolor="black", zorder=5)
ax.axvline(c["peak_H1"]["x"], color=H1_COLOR[63], ls=":", lw=1.3, label="first cycle")
ax.axvline(c["reference"], color="black", ls="--", lw=1.1, label="boundary")
ax.set(xlabel=r"$\Gamma/J$", ylabel="normalized response", xlim=(EXAMPLE_GAMMA_MIN, g.max()), ylim=(0, 1.08))
ax.legend(frameon=False, loc="upper right", fontsize=7.6,
          title=fr"$L={EXAMPLE[0]}$, $h={EXAMPLE[1]}$", title_fontsize=7.6)
ax.grid(alpha=0.2)

# (c) mechanism: the chain closes through its ends
ax = axes[2]
ax.plot(g, c["C_end_end"][show], color="0.2", lw=1.5, label=r"$C_{1,L}$ (end-end)")
ax.plot(g, c["C_end_centre"][show], color=H1_COLOR[63], lw=1.5, ls="-.", label=r"$C_{1,(L+1)/2}$ (end-centre)")
ax.plot(g, c["mz_centre"][show], color="#1b7837", lw=1.3, ls=":", label=r"$\langle\sigma^z_{(L+1)/2}\rangle$ (centre)")
ax.axhline(0.0, color="0.6", lw=0.8)
ax.axvline(c["peak_H1"]["x"], color=H1_COLOR[63], ls=":", lw=1.3)
ax.axvline(c["reference"], color="black", ls="--", lw=1.1)
ax.set(xlabel=r"$\Gamma/J$", ylabel=r"$C_{ij}$, $\langle\sigma^z_i\rangle$", xlim=(EXAMPLE_GAMMA_MIN, g.max()))
ax.legend(frameon=False, loc="center right", fontsize=7.6)
ax.grid(alpha=0.2)

ps.panel_labels(axes)
fig.tight_layout()
ps.save_figure(fig, OUTPUT_DIR, BASENAME)

# %% Derived numbers


def cut_record(L, h, c):
    return {"L": L, "h": h, "chi": c["chi"], "points": int(c["g"].size), "full_gamma": c["full_gamma"],
            "qc_flags": int(c["flag"].sum()), "reference": c["reference"],
            "reference_interpolated": c["reference_interpolated"], "physical": c["peak_physical"],
            "H0": c["peak_H0"], "H1": c["peak_H1"],
            "H1_set": None if c["peak_H1"] is None else ("train" if (L, h) in H1_TRAIN else "test")}


ORDERED_MARGIN = 0.05
ordered_life = []
zz = np.load(DATA_DIR / "clean_frontier_oddL_compact.npz")
for e in json.loads(str(zz["index_json"])):
    p, gg = e["prefix"], np.asarray(zz[e["prefix"] + "_Gamma"], float)
    ivs = np.split(np.asarray(zz[p + "_H1_intervals"], float).reshape(-1, 2), np.cumsum(zz[p + "_H1_counts"])[:-1])
    ref = float(np.interp(e["h"], REF_H, REF_G))
    ordered_life += [float((iv[:, 1] - iv[:, 0]).max()) for gv, iv in zip(gg, ivs) if gv < ref - ORDERED_MARGIN and len(iv)]
even_no_wall = []   # even chains with h < J (no boundary wall): largest H1 lifetime in the whole scan
ze = np.load(DATA_DIR / "clean_frontier_compact.npz")
for (L, h), c in cuts.items():
    if L == max(SIZES) and c["full_gamma"] and h < 1.0:
        p = next(e["prefix"] for e in json.loads(str(ze["index_json"]))
                 if e["L"] == L and abs(e["h"] - h) < 1e-9 and e["chi"] == c["chi"])
        iv = np.asarray(ze[p + "_H1_intervals"], float).reshape(-1, 2)
        even_no_wall.append({"L": L, "h": h, "max_H1_lifetime": float((iv[:, 1] - iv[:, 0]).max()) if len(iv) else 0.0})
ex = odd_cuts[EXAMPLE]
i_on = int(np.argmin(np.abs(ex["g"] - ex["peak_H1"]["x"])))
summary = {
    "figure": BASENAME,
    "rules": {"chi": "highest per (L,h)", "H0_peak": "global max of xi_lambda, QC-masked, local parabola",
              "qc_rule": "exclude points with branch disagreement or bond saturation in themselves or a neighbour",
              "H1": f"first QC-valid interval where N1 rises, N1 = #all H1 lifetimes > {H1_TAU} (ring included), odd L",
              "H1_selection": "amendment v2, PLAN_CLEAN_FRONTIER_ODDL_CLOSURE_20261015.md; train cuts listed in H1_set",
              "H1_complex": "Vietoris-Rips on clip(1-C,0,2), Z2 (gudhi)",
              "controls": "even L: h >= 2(1-1/L) excluded", "not_drawn": [list(k) for k in QC_CUTS],
              "example": {"L": EXAMPLE[0], "h": EXAMPLE[1], "Gamma_min_shown": EXAMPLE_GAMMA_MIN},
              "reference": "digitized boundary of Lajko et al. (2020), linear interpolation between points"},
    "metrics_mae_count": metrics,
    "H1_tau": H1_TAU,
    "odd_h_step": round(float(np.min(np.diff(sorted({h for _, h in odd_cuts})))), 6),
    "ordered_phase_H1": {"margin_below_boundary": ORDERED_MARGIN, "max_lifetime": max(ordered_life)},
    "example_at_H1_onset": {"Gamma": ex["peak_H1"]["x"], "C_end_end": float(ex["C_end_end"][i_on]),
                            "C_end_centre": float(ex["C_end_centre"][i_on])},
    "end_centre_zero_vs_onset": [
        {"L": L, "h": h, "onset": c["peak_H1"]["x"],
         "end_centre_first_negative": (float(c["g"][int(np.argmax(c["C_end_centre"] < 0))])
                                       if (c["C_end_centre"] < 0).any() else None),
         "end_end_at_onset": float(c["C_end_end"][int(np.argmin(np.abs(c["g"] - c["peak_H1"]["x"])))])}
        for (L, h), c in odd_cuts.items() if c["peak_H1"] is not None],
    "factorization_max_abs": {
        "end_end": max(float(np.max(np.abs(c["C_end_end"] - c["mz_end_first"] * c["mz_end_last"]))) for c in odd_cuts.values()),
        "end_centre": max(float(np.max(np.abs(c["C_end_centre"] - (-1) ** ((LL - 1) // 2) * c["mz_end_first"] * c["mz_centre"])))
                          for (LL, _), c in odd_cuts.items())},
    "even_no_wall_H1": even_no_wall,
    "end_magnetization_min_at_onset": min(float(c["mz_end_first"][int(np.argmin(np.abs(c["g"] - c["peak_H1"]["x"])))])
                                          for c in odd_cuts.values() if c["peak_H1"] is not None),
    "centre_sign_change_vs_onset": [
        {"L": L, "h": h, "onset": c["peak_H1"]["x"],
         "centre_mz_sign_change": float(c["g"][int(np.argmax(np.sign(c["mz_centre"]) != np.sign(c["mz_centre"][0])))])}
        for (L, h), c in odd_cuts.items() if c["peak_H1"] is not None],
    "cuts": [cut_record(L, h, c) for (L, h), c in cuts.items()],
    "odd_cuts": [cut_record(L, h, c) for (L, h), c in odd_cuts.items()],
    "cmp_linear_fit": {"form": "Gamma* = a (h_c - h), unweighted, L = 63 cuts with h >= FIT_HMIN",
                       "FIT_HMIN": FIT_HMIN, "reference_slope": CMP_SLOPE, **fits},
    "cmp_cuts": [{**cut_record(L, h, c), "linear_reference": CMP_SLOPE * (2 - h),
                  "H0_slope": c["peak_H0"]["x"] / (2 - h), "physical_slope": c["peak_physical"]["x"] / (2 - h),
                  "H1_slope": None if c["peak_H1"] is None else c["peak_H1"]["x"] / (2 - h),
                  "grid_step": float(np.min(np.diff(c["g"])))}
                 for (L, h), c in sorted(cmp_cuts.items())],
    "low_field_cuts": [{**cut_record(L, h, c),
                        "max_H1_lifetime": c["max_H1_lifetime"],
                        "C_end_end_at_max_Gamma": float(c["C_end_end"][-1])}
                       for (L, h), c in sorted(low_cuts.items())],
}
(OUTPUT_DIR / f"{BASENAME}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
