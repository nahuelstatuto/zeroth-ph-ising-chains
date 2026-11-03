# %% [markdown]
# # How the H0 hierarchy detects the clusters of the strong-disorder RG (V1, didactic chain)
#
# Open chain of L = 12 sites (numbered 1-12 as in Fig. 1), h = 0, chosen couplings (see
# prepare_data.py), exact ground state, C = Cstag, D = 1 - C. The strong-disorder RG decimates
# site 7 (strong field) and joins {4,5,6} and {8,9,10} into one non-contiguous cluster.
# (a) Correlation network of the ground state, drawn as in Fig. 1 (edge width ~ C^2); cluster
#     sites filled.
# (b) H0 hierarchy (single linkage of D): the cluster is one subtree (blue branches).
# (c) Hierarchy restricted to nearest-neighbour links: the two halves meet only through site 7.
# Trees are schematic: branch heights show the order of the merges, not their scale.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent / "_common"))
import paper_style as ps  # noqa: E402

DATA_DIR = HERE / "data_raw"
OUTPUT_DIR = HERE / "output"
BASENAME = "fig_h0_clusters"
CLUSTER = "#0072B2"
EDGE_COLOR = "#8aa4b8"          # as in Fig. 1
SKIPPED = 6                     # 0-based index of site 7

ps.apply_style()
ps.verify_manifest(DATA_DIR)
z = np.load(DATA_DIR / "toy_chain.npz")
C, labels = z["C"], z["sdrg_labels"]
L = C.shape[0]
big = set(np.flatnonzero(labels == labels[4]).tolist())
D = np.clip(1.0 - C, 0.0, 2.0)
np.fill_diagonal(D, 0.0)
Dnn = np.full((L, L), 1.0)
np.fill_diagonal(Dnn, 0.0)
for i in range(L - 1):
    Dnn[i, i + 1] = Dnn[i + 1, i] = D[i, i + 1]

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), gridspec_kw={"width_ratios": (0.85, 1.1, 1.1)})

# %% (a) correlation network, layout of Fig. 1
ax = axes[0]
angles = np.deg2rad(90 - (18 + np.arange(L) * (360 - 36) / (L - 1)))
pos = np.c_[np.cos(angles), np.sin(angles)]
for i in range(L):
    for j in range(i + 1, L):
        w = max(C[i, j], 0.0)
        if w >= 0.05:
            ax.plot(*pos[[i, j]].T, color=EDGE_COLOR, lw=0.3 + 3.2 * w ** 2, alpha=0.25 + 0.75 * w,
                    zorder=1, solid_capstyle="round")
ax.scatter(*pos.T, s=150, color=[CLUSTER if i in big else "white" for i in range(L)],
           edgecolor="0.15", linewidth=0.9, zorder=3)
for k, (x, y) in enumerate(pos):
    ax.text(x, y, str(k + 1), ha="center", va="center", fontsize=7, zorder=4, color="white" if k in big else "black")
xs, ys = pos[SKIPPED]
ax.annotate(r"strong field $\Gamma_7$", xy=(xs, ys), xytext=(xs - 0.15, ys - 0.55), fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal")
ax.axis("off")
ax.set_title("(a) correlation network", loc="left")


# %% (b), (c) schematic trees
def tree(ax, Dm, title, note):
    Z = linkage(squareform(Dm, checks=False), method="single")
    Z[:, 2] = np.arange(1, len(Z) + 1)          # schematic: branch height = merge order (same tree)
    members = {i: {i} for i in range(L)}
    for k, (a, b, _, _) in enumerate(Z):
        members[L + k] = members[int(a)] | members[int(b)]
    dendrogram(Z, ax=ax, link_color_func=lambda node: CLUSTER if members[node] <= big else "0.55",
               labels=[str(i + 1) for i in range(L)], leaf_font_size=9)
    for lbl in ax.get_xticklabels():
        i = int(lbl.get_text()) - 1
        lbl.set_color(CLUSTER if i in big else "0.3")
        lbl.set_fontweight("bold" if i in big else "normal")
    for side in ("left", "top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0)
    ax.set_title(title, loc="left")
    ax.text(0.88, 0.95, note, transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="0.25")


tree(axes[1], D, r"(b) $H_0$ hierarchy", "the cluster is\none subtree")
tree(axes[2], Dnn, "(c) nearest-neighbour links only", "the halves meet\nonly through site 7")
fig.tight_layout()
ps.save_figure(fig, OUTPUT_DIR, BASENAME)
summary = {"cluster_sites_1based": sorted(int(i) + 1 for i in big), "C_6_8": float(C[5, 7]), "C_6_7": float(C[5, 6]),
           "C_7_8": float(C[6, 7]), "D_merge_cluster_H0": float(1 - C[5, 7]), "D_merge_halves_NN": float(1 - C[5, 6])}
(OUTPUT_DIR / f"{BASENAME}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
