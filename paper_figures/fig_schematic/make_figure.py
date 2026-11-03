# %% [markdown]
# # Schematic: from one quantum state to the landscape velocity
#
# One realization (random h=0, OBC, L=16, seed 43, chi=64) followed through
# the whole pipeline:
# (a) correlation network of the state, weights C = Cstag;
# (a1-a4) filtration of D = 1 - C: edges enter when D_ij <= eps, components
#     merge (orange edges = merges = H0 deaths);
# (b) H0 barcode, dotted lines = the four snapshots;
# (c) persistence landscape: each bar [0, d] becomes a tent of height d/2;
# (d) the landscape moves when Gamma0 changes;
# (e) landscape velocity xi_lambda(Gamma0) of this same realization.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent / "_common"))
import paper_style as ps  # noqa: E402

DATA_DIR = HERE / "data_raw"
OUTPUT_DIR = HERE / "output"
BASENAME = "fig_schematic"
STATE_FILE = "random_h0_L16_seed43_chi64.npz"
GAMMA_STATE = 0.95          # state shown in (a)-(c)
GAMMA_PAIR = (0.85, 0.95)   # two fields compared in (d)
SNAPSHOT_COMPONENTS = (12, 7, 3, 1)
T_GRID = np.linspace(0.0, 1.0, 2001)
TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

EDGE_COLOR = "#8aa4b8"      # network edges that do not merge components
MERGE_COLOR = ps.TOPOLOGY_COLOR
FAR_COLOR = "#0072B2"         # V1: merges between sites at distance |i-j| >= 2 (as the clusters of Fig. 4)
EARLY_COLOR = "0.55"        # landscape at the lower field in (d)

ps.apply_style()
ps.verify_manifest(DATA_DIR)

# %% Data and H0 by single linkage (Kruskal)
with np.load(DATA_DIR / STATE_FILE) as z:
    gammas = np.asarray(z["Gamma0_values"], float)
    cstag = np.asarray(z["Cstag"], float)
    qc_flags = np.asarray(z["branch_disagreement"], bool)
L = cstag.shape[1]
if qc_flags.any():
    raise ValueError("The schematic state should have no branch-disagreement flags")


def h0_merges(C):
    """Return the merge edges (i, j, death) of the H0 filtration of D = 1 - C."""
    D = 1.0 - C
    pairs = sorted((D[i, j], i, j) for i in range(L) for j in range(i + 1, L))
    parent = list(range(L))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    merges = []
    for d, i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            merges.append((i, j, float(d)))
    return merges


def tents(deaths):
    layers = np.maximum(0.0, np.minimum(T_GRID[None, :], np.asarray(deaths)[:, None] - T_GRID[None, :]))
    return np.sort(layers, axis=0)[::-1]


def gidx(value):
    return int(np.argmin(np.abs(gammas - value)))


deaths_all = np.array([[m[2] for m in h0_merges(C)] for C in cstag])
layers_all = np.stack([tents(d) for d in deaths_all])
d_layers = np.gradient(layers_all, gammas, axis=0, edge_order=2)
xi = np.sqrt(TRAPEZOID((d_layers ** 2).sum(axis=1), T_GRID, axis=1) / (L - 1))
if deaths_all.max() >= T_GRID[-1]:
    raise ValueError("Landscape grid does not cover all deaths")

i_state = gidx(GAMMA_STATE)
C_state = cstag[i_state]
merges = h0_merges(C_state)
deaths = np.array([m[2] for m in merges])
ordered = np.sort(deaths)
snap_eps = []
for n_comp in SNAPSHOT_COMPONENTS:
    k = L - n_comp                       # merges done
    upper = ordered[k] if k < ordered.size else ordered[-1] + 0.06
    snap_eps.append(0.5 * (ordered[k - 1] + upper))

# %% Drawing helpers
angles = np.deg2rad(90 - (18 + np.arange(L) * (360 - 36) / (L - 1)))
pos = np.c_[np.cos(angles), np.sin(angles)]
CIRCLED = ["(i)", "(ii)", "(iii)", "(iv)"]


def draw_network(ax, eps=None):
    D = 1.0 - C_state
    merge_set = {(min(i, j), max(i, j)) for i, j, d in merges if eps is None or d <= eps}
    for i in range(L):
        for j in range(i + 1, L):
            if eps is None:
                w = max(C_state[i, j], 0.0)
                if w < 0.05:
                    continue
                ax.plot(*pos[[i, j]].T, color=EDGE_COLOR, lw=0.3 + 3.2 * w ** 2,
                        alpha=0.25 + 0.75 * w, zorder=1, solid_capstyle="round")
            elif D[i, j] <= eps:
                is_merge = (i, j) in merge_set
                color = (FAR_COLOR if j - i > 1 else MERGE_COLOR) if is_merge else EDGE_COLOR
                ax.plot(*pos[[i, j]].T, color=color,
                        lw=2.2 if is_merge else 0.8, zorder=2 if is_merge else 1,
                        solid_capstyle="round")
    ax.scatter(*pos.T, s=95, color="white", edgecolor="0.15", linewidth=0.9, zorder=3)
    for k, (x, y) in enumerate(pos):
        ax.text(x, y, str(k + 1), ha="center", va="center", fontsize=6.2, zorder=4)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")


def flow_arrow(fig, ax_from, ax_to, label):
    b1, b2 = ax_from.get_position(), ax_to.get_position()
    y = b1.y0 + 0.5 * (b1.y1 - b1.y0)
    fig.add_artist(FancyArrowPatch((b1.x1 + 0.006, y), (b2.x0 - 0.048, y), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=13, color="0.3", lw=1.2))
    fig.text(0.5 * (b1.x1 + b2.x0) - 0.021, y + 0.02, label, ha="center", va="bottom",
             fontsize=8.6, color="0.25")

# %% Figure
fig = plt.figure(figsize=(13.2, 6.9))
outer = fig.add_gridspec(2, 1, height_ratios=(1.0, 1.08), hspace=0.42,
                         left=0.045, right=0.99, bottom=0.08, top=0.88)
top = outer[0].subgridspec(1, 5, width_ratios=(1.25, 1, 1, 1, 1), wspace=0.05)
bottom = outer[1].subgridspec(1, 4, width_ratios=(1.0, 1.0, 1.0, 1.0), wspace=0.62)

ax_net = fig.add_subplot(top[0])
draw_network(ax_net)
ax_net.set_title(r"(a) correlation network of one state", loc="left", fontsize=9.5)
ax_net.text(0, -1.38, fr"edge width $\propto C_{{ij}}=(-1)^{{|i-j|}}\langle\sigma^z_i\sigma^z_j\rangle$"
            "\n" fr"random $h=0$, $L={L}$, $\Gamma_0={GAMMA_STATE}$", ha="center", va="top", fontsize=8.6)

snap_axes = []
for k, eps in enumerate(snap_eps):
    ax = fig.add_subplot(top[k + 1])
    draw_network(ax, eps)
    n_comp = L - int(np.sum(deaths <= eps))
    ax.set_title(fr"{CIRCLED[k]}  $\varepsilon={eps:.2f}$" + "\n" + f"{n_comp} component" + ("s" if n_comp > 1 else ""),
                 fontsize=8.6)
    snap_axes.append(ax)
fig.text(snap_axes[0].get_position().x0 + 0.02, 0.975,
         r"filtration of $D=1-C$: an edge enters when $D_{ij}\leq\varepsilon$"
         r"  (orange or blue = edge that merges two components, blue when $|i-j|\geq2$)", fontsize=8.6)

# (b) barcode
ax_bar = fig.add_subplot(bottom[0])
order = np.argsort(deaths)[::-1]
t_max = 0.72
for y, idx in enumerate(order):
    far = abs(merges[idx][0] - merges[idx][1]) > 1
    ax_bar.hlines(y + 1, 0, deaths[idx], color=FAR_COLOR if far else MERGE_COLOR, lw=2.0)
ax_bar.annotate("", xy=(t_max, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=MERGE_COLOR, lw=2.0, mutation_scale=10))
for k, eps in enumerate(snap_eps):
    ax_bar.axvline(eps, ymin=0, ymax=0.95, color="0.35", ls=":", lw=1.0)
    ax_bar.text(eps, L + 0.9, CIRCLED[k], ha="center", va="bottom", fontsize=9)
ax_bar.set(xlim=(0, t_max), ylim=(-0.8, L + 1.6), xlabel=r"$\varepsilon$")
ax_bar.set_yticks([])
ax_bar.spines["left"].set_visible(False)
ax_bar.set_title(r"(b) $H_0$ barcode", loc="left", fontsize=9.5, pad=14)

# (c) landscape of the same state
ax_land = fig.add_subplot(bottom[1])
lam = layers_all[i_state]
for k in range(L - 1)[::-1]:
    ax_land.fill_between(T_GRID, 0, lam[k], color=MERGE_COLOR, alpha=0.06, lw=0)
    ax_land.plot(T_GRID, lam[k], color=MERGE_COLOR, lw=0.6 if k > 2 else 1.6 - 0.35 * k,
                 alpha=1.0 if k <= 2 else 0.55)
for k in range(3):
    peak = np.argmax(lam[k])
    ax_land.text(T_GRID[peak] - 0.02, lam[k][peak] + 0.004, fr"$\lambda_{k + 1}$", ha="center", va="bottom",
                 fontsize=7.6)
ax_land.set(xlim=(0, t_max), ylim=(0, 0.34), xlabel=r"$t$", ylabel=r"$\lambda_k(t)$")
ax_land.text(0.98, 0.82, "bar $[0,d]\\mapsto$ tent\nof height $d/2$;\n$\\lambda_k$: $k$-th largest",
             transform=ax_land.transAxes, ha="right", va="top", fontsize=7.4)
ax_land.set_title(r"(c) persistence landscape", loc="left", fontsize=9.5, pad=14)

# (d) motion of the landscape with the field
ax_move = fig.add_subplot(bottom[2])
la, lb = layers_all[gidx(GAMMA_PAIR[0])], layers_all[gidx(GAMMA_PAIR[1])]
for k in range(L - 1):
    ax_move.plot(T_GRID, la[k], color=EARLY_COLOR, lw=0.9 if k else 1.4, ls="--")
    ax_move.plot(T_GRID, lb[k], color=MERGE_COLOR, lw=0.9 if k else 1.4)
ax_move.fill_between(T_GRID, la[0], lb[0], color=MERGE_COLOR, alpha=0.18, lw=0)
ax_move.legend(handles=[plt.Line2D([], [], color=EARLY_COLOR, ls="--", label=fr"$\Gamma_0={GAMMA_PAIR[0]}$"),
                        plt.Line2D([], [], color=MERGE_COLOR, label=fr"$\Gamma_0={GAMMA_PAIR[1]}$")],
               frameon=False, loc="upper right", fontsize=7.6)
ax_move.set(xlim=(0, t_max), ylim=(0, 0.34), xlabel=r"$t$", ylabel=r"$\lambda_k(t)$")
ax_move.set_title(r"(d) the landscape moves with $\Gamma_0$", loc="left", fontsize=9.5, pad=14)

# (e) landscape velocity of this realization
ax_xi = fig.add_subplot(bottom[3])
ax_xi.plot(gammas, xi, color=MERGE_COLOR, lw=1.6, marker="o", ms=2.5)
peak = ps.quadratic_peak(gammas, xi)
ax_xi.scatter(peak["x"], peak["value"], marker="s", s=45, color=MERGE_COLOR, edgecolor="black", zorder=5,
              label=fr"$\Gamma^\star_0={peak['x']:.3f}$")
ax_xi.axvspan(*GAMMA_PAIR, color=MERGE_COLOR, alpha=0.12, lw=0)
ax_xi.axvline(1.0, color="black", ls="--", lw=0.9)
ax_xi.set(xlabel=r"$\Gamma_0$", ylabel=r"$\xi_\lambda$", ylim=(0, 1.32 * peak["value"]))
ax_xi.legend(frameon=False, loc="upper right", fontsize=7.6)
ax_xi.set_title(r"(e) landscape velocity", loc="left", fontsize=9.5, pad=14)
#ax_xi.text(0.03, 0.97, r"$\xi_\lambda^2=\frac{1}{L-1}\sum_k\int dt\,|\partial_{\Gamma_0}\lambda_k|^2$",
#           transform=ax_xi.transAxes, ha="left", va="top", fontsize=8.2)

for a, b, label in ((ax_bar, ax_land, "bar → tent"), (ax_land, ax_move, r"vary $\Gamma_0$"),
                    (ax_move, ax_xi, r"$\|\partial_{\Gamma_0}\lambda\|_2$")):
    flow_arrow(fig, a, b, label)

ps.save_figure(fig, OUTPUT_DIR, BASENAME)

summary = {
    "figure": BASENAME,
    "state": {"file": STATE_FILE, "L": L, "Gamma0": float(gammas[i_state])},
    "snapshot_eps": [float(e) for e in snap_eps],
    "snapshot_components": list(SNAPSHOT_COMPONENTS),
    "deaths_at_state": sorted(float(d) for d in deaths),
    "non_nearest_neighbour_merges": [(int(i) + 1, int(j) + 1) for i, j, _ in merges if abs(i - j) > 1],
    "xi_peak": peak,
}
(OUTPUT_DIR / f"{BASENAME}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary["non_nearest_neighbour_merges"]), summary["snapshot_eps"], peak)
