# %% [markdown]
# # Fig. S3 (V1): the first H1 cycle in one cut
#
# Clean odd chain L = 31, h = 1.7 J (OBC, chi = 128), data_raw/clean_frontier_h1_cycle_L31.npz.
# Vietoris-Rips persistence of D = 1 - C up to dimension 2 (GUDHI). For the longest H1 interval,
# the representative cycle is its birth edge (u, v) closed by the shortest path from u to v in the
# graph of edges with D <= birth (Dijkstra with weights D, birth edge removed).
# (a), (b) Sites placed by classical multidimensional scaling of D, for display only (the
# persistence is computed on D). Grey lines are the edges with D <= eps, with eps the midpoint of the
# cycle's lifetime in (b), the same in both panels. (c) Lifetimes of the two longest H1 intervals
# against Gamma, with the threshold tau of the paper and the onset rule of Sec. V.

# %%
import json
import sys
from pathlib import Path

import gudhi
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import dijkstra

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent / "_common"))
import paper_style as ps  # noqa: E402

DATA_DIR = HERE / "data_raw"
OUTPUT_DIR = HERE / "output"
BASENAME = "fig_clean_frontier_S3_h1_cycle"
REF_H = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.9, 1.95, 2.0])
REF_G = np.array([1.0, 0.992, 0.970, 0.931, 0.878, 0.802, 0.712, 0.599, 0.453, 0.278, 0.154, 0.077, 0.0])
TAU = 0.04                     # threshold of the paper (Sec. V)
GAMMA_PANELS = (0.40, 0.50)    # one point before the onset, one after it
CYCLE_COLOR = ps.TOPOLOGY_COLOR

ps.apply_style()
ps.verify_manifest(DATA_DIR)
z = np.load(DATA_DIR / "clean_frontier_h1_cycle_L31.npz")
g, C = z["Gamma"], z["Cstag"]
L, h = int(z["L"]), float(z["h"])
flag = z["branch_disagreement"] | (z["max_bond"] >= int(z["chi"]))


# %% Persistence and representative cycle
def dissimilarity(k):
    D = 1.0 - C[k]
    np.fill_diagonal(D, 0.0)
    return D


def h1_pairs(D):
    st = gudhi.RipsComplex(distance_matrix=D, max_edge_length=2.0).create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    pairs = [(st.filtration(b), st.filtration(d) if d else np.inf, tuple(b), tuple(d))
             for b, d in st.persistence_pairs() if len(b) == 2]
    return sorted(pairs, key=lambda p: p[1] - p[0], reverse=True)


def cycle(D, birth, u, v):
    W = np.where(D <= birth + 1e-12, D, 0.0)
    W[u, v] = W[v, u] = 0.0
    _, pred = dijkstra(W, indices=u, return_predecessors=True)
    path = [v]
    while path[-1] != u:
        path.append(int(pred[path[-1]]))
    return path


def mds(D):
    n = len(D)
    J = np.eye(n) - 1.0 / n
    w, V = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)
    X = V[:, -2:][:, ::-1] * np.sqrt(np.maximum(w[-2:][::-1], 0.0))
    X = X[:, ::-1]                                   # second axis vertical
    if X[L // 2, 1] < X[0, 1]:                       # centre on top
        X[:, 1] *= -1
    if X[L // 4, 0] > X[3 * L // 4, 0]:              # left half on the left
        X[:, 0] *= -1
    return X


life = np.zeros((len(g), 2))
for k in range(len(g)):
    p = h1_pairs(dissimilarity(k))
    for r in range(min(2, len(p))):
        life[k, r] = p[r][1] - p[r][0]
n1 = (life > TAU).sum(1)          # enough here: no third interval reaches tau in this cut
rise = (np.diff(n1) >= 1) & ~(flag[:-1] | flag[1:])
onset = float(0.5 * (g[np.flatnonzero(rise)[0]] + g[np.flatnonzero(rise)[0] + 1]))
reference = float(np.interp(h, REF_H, REF_G))

kb = int(np.argmin(abs(g - GAMMA_PANELS[1])))
Db = dissimilarity(kb)
birth, death, (u, v), tri = h1_pairs(Db)[0]       # birth edge (u, v), triangle that kills the cycle
eps = 0.5 * (birth + death)
loop = cycle(Db, birth, u, v)
summary = {"L": L, "h": h, "tau": TAU, "onset": onset, "reference": reference,
           "gamma_panels": [float(g[int(np.argmin(abs(g - x)))]) for x in GAMMA_PANELS],
           "eps": eps, "birth": birth, "death": death, "lifetime": death - birth,
           "birth_edge_sites": [u + 1, v + 1], "death_triangle_sites": sorted(s + 1 for s in tri),
           "cycle_sites": [s + 1 for s in loop],
           "cycle_sites_on_centre_sublattice": int(sum((s % 2) == (L // 2) % 2 for s in loop)),
           "D_end_end": float(Db[0, L - 1]), "D_end_centre": float(Db[0, L // 2]),
           "max_lifetime_before_onset": float(life[g < onset].max())}

# %% Figure
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), gridspec_kw={"width_ratios": [1, 1, 1.25]})
same = np.arange(L) % 2 == 0                      # sublattice of the ends (sites 1, 3, ..., L)
for ax, gp in zip(axes[:2], GAMMA_PANELS):
    k = int(np.argmin(abs(g - gp)))
    D = dissimilarity(k)
    X = mds(D)
    for i in range(L):
        for j in range(i + 1, L):
            if D[i, j] <= eps:
                ax.plot(*X[[i, j]].T, color="0.82", lw=0.4, zorder=1)
    if k == kb:
        for i, j in zip(loop, loop[1:] + loop[:1]):
            ax.plot(*X[[i, j]].T, color=CYCLE_COLOR, lw=1.8, zorder=2, solid_capstyle="round")
        ax.plot(*X[[u, v]].T, color="0.1", lw=1.3, ls=(0, (2, 1.2)), zorder=2.5)          # birth edge
        ax.add_patch(plt.Polygon(X[list(tri)], closed=True, facecolor=CYCLE_COLOR, alpha=0.3,
                                 edgecolor=CYCLE_COLOR, lw=0.8, zorder=1.5))             # death triangle
        ax.scatter(*X[list(tri)].T, s=55, facecolor="none", edgecolor=CYCLE_COLOR, lw=1.0, zorder=4)
        ax.annotate("birth", X[[u, v]].mean(0), xytext=(8, 0), textcoords="offset points",
                    fontsize=6.5, ha="left", va="center")
        ax.annotate("death", X[list(tri)].mean(0), xytext=(-8, 0), textcoords="offset points",
                    fontsize=6.5, ha="right", va="center", color=CYCLE_COLOR)
    ax.scatter(*X[same].T, s=14, color="0.15", zorder=3, label="sublattice of the ends")
    ax.scatter(*X[~same].T, s=14, facecolor="white", edgecolor="0.15", lw=0.8, zorder=3,
               label="opposite sublattice")
    for s, off, va in ((0, (-6, -4), "top"), (L // 2, (0, 4), "bottom"), (L - 1, (6, -4), "top")):
        ax.annotate(str(s + 1), X[s], xytext=off, textcoords="offset points", fontsize=7,
                    ha="center", va=va)
    ax.set_title(rf"$\Gamma={g[k]:.2f}J$", fontsize=8, pad=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1)

ax = axes[2]
ax.plot(g, life[:, 0], "-", color=CYCLE_COLOR, lw=1.4, label="longest")
ax.plot(g, life[:, 1], "--", color=CYCLE_COLOR, lw=1.0, alpha=0.6, label="second")
ax.axhline(TAU, color="0.3", ls=":", lw=0.9)
ax.text(0.02, TAU + 0.004, r"$\tau$", fontsize=8, color="0.3")
ax.axvline(reference, color="0.3", ls="--", lw=0.9)
ax.axvline(onset, color=CYCLE_COLOR, ls=":", lw=0.9)
for gp, lab in zip(summary["gamma_panels"], "ab"):
    k = int(np.argmin(abs(g - gp)))
    ax.plot(g[k], life[k, 0], "o", ms=3.5, color=CYCLE_COLOR, mfc="white", mew=1.0, zorder=4)
    ax.annotate(f"({lab})", (g[k], life[k, 0]), xytext=(5, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7)
ax.set(xlabel=r"$\Gamma/J$", ylabel=r"$H_1$ lifetime", xlim=(g.min(), g.max()))
ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=7, loc="upper right")
ps.panel_labels(axes)
fig.tight_layout(w_pad=0.6)
ps.save_figure(fig, OUTPUT_DIR, BASENAME)
(OUTPUT_DIR / f"{BASENAME}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=1))
