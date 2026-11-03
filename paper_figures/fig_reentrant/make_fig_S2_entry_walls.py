# %% [markdown]
# # Fig. S2 (V1): domain walls across the reentrant entry
#
# OBC, fixed_mean, L = 32, h = 2.3125. Staggered local magnetization (-1)^i <sigma^z_i> of three
# reentrant realizations at the two grid points that bracket their entry (open symbols before,
# filled after). Domain walls (nearest-neighbour pairs aligned along z) are marked by vertical
# lines (dotted before, solid after); the shaded band is the block whose staggered orientation
# reverses at the entry.
# (a) Delta M = -2: a block of odd length reverses and two walls annihilate.
# (b) Delta M = 0: a block of even length reverses and the wall is displaced.
# (c) Delta M = 0: a block of even length reverses and two walls annihilate.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent / "_common"))
import paper_style as ps  # noqa: E402

DATA_DIR = HERE / "data_raw"
OUTPUT_DIR = HERE / "output"
BASENAME = "fig_reentrant_S2_entry_walls"
MZ_MIN = 0.2
BEFORE, AFTER = "0.55", "#0072B2"

ps.apply_style()
ps.verify_manifest(DATA_DIR)
z = np.load(DATA_DIR / "reentrant_entry_walls.npz")


def walls(m):
    ok = (np.abs(m[:-1]) >= MZ_MIN) & (np.abs(m[1:]) >= MZ_MIN)
    return np.flatnonzero(ok & (m[:-1] * m[1:] > 0))


fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6), sharey=True)
titles = {"odd": r"(a) $\Delta M=-2$, odd block, two walls annihilate",
          "disp": r"(b) $\Delta M=0$, even block, wall displaced",
          "even": r"(c) $\Delta M=0$, even block, two walls annihilate"}
summary = {}
for ax, name in zip(axes, ("odd", "disp", "even")):
    mb, ma = z[f"{name}_before"], z[f"{name}_after"]
    L = mb.size
    x = np.arange(1, L + 1)
    sign = (-1) ** np.arange(L)
    sb, sa = sign * mb, sign * ma
    flipped = (np.abs(mb) >= MZ_MIN) & (np.abs(ma) >= MZ_MIN) & (np.sign(sb) != np.sign(sa))
    for i in np.flatnonzero(flipped):
        ax.axvspan(x[i] - 0.5, x[i] + 0.5, color="#E69F00", alpha=0.18, lw=0)
    for w in walls(mb):
        ax.axvline(x[w] + 0.5, color=BEFORE, ls=":", lw=1.1)
    for w in walls(ma):
        ax.axvline(x[w] + 0.5, color=AFTER, ls="-", lw=1.1)
    g0, g1 = z[f"{name}_gamma"]
    ax.plot(x, sb, "o", mfc="white", mec=BEFORE, ms=4.5, label=fr"before, $\Gamma_0={g0:.3f}$")
    ax.plot(x, sa, "o", color=AFTER, ms=4.0, label=fr"after, $\Gamma_0={g1:.3f}$")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set(xlabel="site $i$", xlim=(0.5, L + 0.5), ylim=(-1.15, 1.15), title=titles[name])
    ax.title.set_ha("left")
    ax.title.set_position((0.0, 1.0))
    ax.legend(frameon=False, fontsize=7.4, loc="center left")
    summary[name] = {"seed": int(z[f"{name}_seed"]), "dM": float(z[f"{name}_dM"]),
                     "walls_before_after": z[f"{name}_walls"].tolist(), "reversed_sites": int(flipped.sum())}
axes[0].set_ylabel(r"$(-1)^i\langle\sigma^z_i\rangle$")
fig.tight_layout()
ps.save_figure(fig, OUTPUT_DIR, BASENAME)
summary["class_statistics"] = json.loads(str(z["stats_json"]))
(OUTPUT_DIR / f"{BASENAME}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
