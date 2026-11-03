# %% [markdown]
# # Fig. S4 (V1): robustness of the first-H1-cycle onset to the lifetime threshold tau
#
# Clean odd chains L = 31 and 63 in a longitudinal field (OBC), the nine cuts h = 1.1-1.9 of
# Fig. 5 (data_raw/clean_frontier_oddL_compact.npz, same QC mask: an increase of N1 counts only
# between two unflagged points). For each threshold tau the onset is the midpoint of the first
# field interval where the number N1(tau) of H1 intervals with lifetime above tau increases (rule
# of the paper, tau = 0.04 there). Shown: mean and largest absolute deviation of the onset from the
# reference boundary, against tau, and the number of cuts without any onset.

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
BASENAME = "fig_clean_frontier_S4_h1_tau"
REF_H = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.9, 1.95, 2.0])
REF_G = np.array([1.0, 0.992, 0.970, 0.931, 0.878, 0.802, 0.712, 0.599, 0.453, 0.278, 0.154, 0.077, 0.0])
GRID_H = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9}
TAUS = np.round(np.arange(0.005, 0.1501, 0.005), 3)
TAU_PAPER = 0.04
COLORS = {63: "#762a83", 31: "#c2a5cf"}

ps.apply_style()
ps.verify_manifest(DATA_DIR)
z = np.load(DATA_DIR / "clean_frontier_oddL_compact.npz")
best = {}
for e in json.loads(str(z["index_json"])):
    key = (e["L"], round(e["h"], 4))
    if round(e["h"], 4) in GRID_H and (key not in best or e["chi"] > best[key]["chi"]):
        best[key] = e


def onset(g, counts, flat, flag, tau):
    intervals = np.split(flat.reshape(-1, 2), np.cumsum(counts)[:-1])
    n1 = np.array([int(np.sum(iv[:, 1] - iv[:, 0] > tau)) if len(iv) else 0 for iv in intervals])
    rise = (np.diff(n1) >= 1) & ~(flag[:-1] | flag[1:])
    return None if not rise.any() else float(0.5 * (g[np.flatnonzero(rise)[0]] + g[np.flatnonzero(rise)[0] + 1]))


summary = {"taus": TAUS.tolist(), "tau_paper": TAU_PAPER}
fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.8))
for L in (63, 31):
    mae, mx, missing = [], [], []
    for tau in TAUS:
        errs, miss = [], 0
        for (LL, h), e in best.items():
            if LL != L:
                continue
            p = e["prefix"]
            g = np.asarray(z[f"{p}_Gamma"], float)
            flag = np.asarray(z[f"{p}_branch_disagreement"], bool) | (np.asarray(z[f"{p}_max_bond"]) >= e["chi"])
            x = onset(g, np.asarray(z[f"{p}_H1_counts"], int), np.asarray(z[f"{p}_H1_intervals"], float), flag, tau)
            if x is None:
                miss += 1
            else:
                errs.append(abs(x - float(np.interp(h, REF_H, REF_G))))
        mae.append(np.mean(errs) if errs else np.nan)
        mx.append(np.max(errs) if errs else np.nan)
        missing.append(miss)
    mae, mx = np.array(mae), np.array(mx)
    ax.plot(TAUS, mae, "o-", color=COLORS[L], ms=4, lw=1.4, label=fr"$L={L}$, mean")
    ax.plot(TAUS, mx, "s--", color=COLORS[L], ms=3, lw=1.0, mfc="white", label=fr"$L={L}$, largest")
    ok = (np.array(missing) == 0) & (mae <= 2 * mae[np.argmin(np.abs(TAUS - TAU_PAPER))])
    summary[f"L{L}"] = {"mae": mae.tolist(), "max": mx.tolist(), "missing": missing,
                        "mae_at_paper_tau": float(mae[np.argmin(np.abs(TAUS - TAU_PAPER))]),
                        "tau_range_within_twice_paper_mae_and_no_missing": [float(TAUS[ok].min()), float(TAUS[ok].max())] if ok.any() else None}
ax.axvline(TAU_PAPER, color="0.3", ls=":", lw=1.0)
ax.text(TAU_PAPER + 0.002, 0.97, r"$\tau$ of the paper", transform=ax.get_xaxis_transform(), fontsize=8, va="top")
ax.set(xlabel=r"lifetime threshold $\tau$", ylabel=r"deviation of the onset from the boundary", yscale="log")
ax.legend(frameon=False, fontsize=7.6, ncol=2, loc="lower right")
fig.tight_layout()
ps.save_figure(fig, OUTPUT_DIR, BASENAME)
(OUTPUT_DIR / f"{BASENAME}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({L: {k: summary[f"L{L}"][k] for k in ("mae_at_paper_tau", "tau_range_within_twice_paper_mae_and_no_missing")} for L in (31, 63)}))
