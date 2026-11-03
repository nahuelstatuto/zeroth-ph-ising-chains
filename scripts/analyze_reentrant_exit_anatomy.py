"""Anatomy of the reentrant entry and exit, and what topology sees (OBC L16 / L32).

Exploratory, no DMRG. Inputs (all produced read-only from frozen data):
  reentrant-exit-anatomy-v1/analysis/cache_L{16,32}.npz       (build_reentrant_exit_anatomy_cache.py)
  reentrant-exit-anatomy-v1/analysis/local_magnetization_L{16,32}.npz  (measure_reentrant_local_magnetization.py)
  L16/L32 all-1000 analysis_arrays.npz                        (frozen H0 landscape velocity)

Physical selection, entry/exit crossings and H0 velocity are the frozen ones.
Outputs: summary.json, per-seed CSV and two exploratory figures in the same folder.
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "random_ising_gamma_ensemble"
OUT = BASE / "reentrant-exit-anatomy-v1" / "analysis"
FROZEN = {16: "L16-reentrant-all1000-complete-v1", 32: "L32-reentrant-h23125-all1000-complete-v1"}
H1_TAU = 0.006
N_PERM = 2000
RNG = np.random.default_rng(20261013)


# ----------------------------------------------------------------------------- helpers
def crossings(g, c):
    out = []
    for i in range(len(g) - 1):
        a, b = c[i], c[i + 1]
        if a * b < 0 or (b == 0 and a != 0):
            out.append((g[i] + (g[i + 1] - g[i]) * a / (a - b), (b - a) / (g[i + 1] - g[i])))
    return out


def entry_exit(g, c):
    cr = crossings(g, c)
    up = [x for x in cr if x[1] > 0]
    dn = [x for x in cr if x[1] < 0]
    e = up[0] if up else (np.nan, np.nan)
    x = dn[-1] if dn and up and dn[-1][0] > up[0][0] else (np.nan, np.nan)
    return e, x


def pair_curve(C, r):
    return np.diagonal(C, offset=r, axis1=-2, axis2=-1).mean(-1)


def mst_deaths(M):
    L = M.shape[0]
    D = 1.0 - M
    iu = np.triu_indices(L, 1)
    order = np.argsort(D[iu], kind="stable")
    parent = list(range(L))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    out = []
    for k in order:
        i, j = iu[0][k], iu[1][k]
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            out.append(D[i, j])
            if len(out) == L - 1:
                break
    return np.sort(out)


def landscape_velocity(g, stack):
    L = stack.shape[-1]
    deaths = np.array([mst_deaths(M) for M in stack])
    t = np.linspace(0.0, max(2.0, 1.02 * deaths.max()), 801)
    lam = np.sort(np.maximum(0, np.minimum(t[None, None, :], deaths[:, :, None] - t[None, None, :])), axis=1)
    return np.sqrt(np.trapezoid((np.gradient(lam, g, axis=0) ** 2).sum(1), t, axis=1) / (L - 1))


def square_gaps(M):
    L = M.shape[0]
    s = np.arange(L - 3)
    return np.minimum.reduce([M[s, s + 1], M[s, s + 2], M[s + 1, s + 3], M[s + 2, s + 3]]) - np.maximum(M[s + 1, s + 2], M[s, s + 3])


def localization(est, ref, step):
    ok = np.isfinite(est) & np.isfinite(ref)
    e, r = est[ok], ref[ok]
    perm = np.array([np.mean(np.abs(e - RNG.permutation(r))) for _ in range(N_PERM)])
    mae = float(np.mean(np.abs(e - r)))
    return {"n": int(ok.sum()), "MAE": mae, "constant_median_MAE": float(np.mean(np.abs(np.median(r) - r))),
            "one_step": float(np.mean(np.abs(e - r) <= step + 1e-9)), "spearman": float(spearmanr(e, r).statistic),
            "perm_p": float((np.sum(perm <= mae) + 1) / (N_PERM + 1))}


def aligned_bump(v, x, ref, step, after=None, win=1, far=(3, 6)):
    """Mean log-ratio of v near each seed's reference vs its 3..6-step surroundings."""
    vals = []
    for n in range(len(ref)):
        dx = (x - ref[n]) / step
        ok = np.ones_like(x, bool) if after is None else x > after[n] + 2 * step
        a = ok & (np.abs(dx) <= win + 1e-9)
        b = ok & (np.abs(dx) >= far[0] - 1e-9) & (np.abs(dx) <= far[1] + 1e-9)
        if a.any() and b.any():
            vals.append(np.log(v[n][a].mean() / v[n][b].mean()))
    return float(np.mean(vals))


# ----------------------------------------------------------------------------- per size
def analyse(L):
    cache = np.load(OUT / f"cache_L{L}.npz")
    mag = np.load(OUT / f"local_magnetization_L{L}.npz")
    frozen = np.load(BASE / FROZEN[L] / "analysis" / "analysis_arrays.npz")
    g = cache["Gamma0"]
    step = float(g[1] - g[0])
    sel = np.flatnonzero(cache["selected"])
    if not np.array_equal(sel, mag["sel"]):
        raise ValueError("magnetization file does not match the selection")
    C = cache["Cstag"][sel].astype(float)
    R = L // 2 - 1
    entry, exit_ = cache["entry"][sel], cache["exit"][sel]
    old = np.array([c.startswith("old") for c in cache["cohort"][sel]])
    res = {"L": L, "h": float({16: 2.5, 32: 2.3125}[L]), "selected": int(sel.size), "R": R}

    # 1. sharpness and scale dependence of the physical crossings
    slopes_e, slopes_x, amp = [], [], []
    for n in range(sel.size):
        cv = pair_curve(C[n], R)
        (e, se), (x, sx) = entry_exit(g, cv)
        slopes_e.append(se)
        slopes_x.append(abs(sx))
        amp.append(cv.max())
    slopes_e, slopes_x, amp = map(np.asarray, (slopes_e, slopes_x, amp))
    res["crossing_slope_median"] = {"entry": float(np.median(slopes_e)), "exit": float(np.median(slopes_x))}
    res["crossing_width_median"] = {"entry": float(np.median(amp / slopes_e)), "exit": float(np.median(amp / slopes_x))}
    res["fraction_entry_steeper_than_exit"] = float(np.mean(slopes_e > slopes_x))
    scale = []
    for r in range(3, L - 2, 2):
        ee, xx = [], []
        for n in range(sel.size):
            (e, _), (x, _) = entry_exit(g, pair_curve(C[n], r))
            ee.append(e)
            xx.append(x)
        ee, xx = np.asarray(ee), np.asarray(xx)
        scale.append({"r": r, "n_entry": int(np.isfinite(ee).sum()), "median_entry": float(np.nanmedian(ee)) if np.isfinite(ee).any() else None,
                      "median_abs_entry_shift": float(np.nanmedian(np.abs(ee - entry))) if np.isfinite(ee).any() else None,
                      "n_exit": int(np.isfinite(xx).sum()), "median_exit": float(np.nanmedian(xx)) if np.isfinite(xx).any() else None,
                      "median_abs_exit_shift": float(np.nanmedian(np.abs(xx - exit_))) if np.isfinite(xx).any() else None})
    res["scale_dependence"] = scale

    # 2. magnetization decomposition, first-order-like signatures
    sig = mag["mz"] * ((-1) ** np.arange(L))[None, None, :]
    dis = sig[..., :, None] * sig[..., None, :]
    Dr, Kr = pair_curve(dis, R), pair_curve(C - dis, R)
    ms, mu = np.abs(sig.mean(-1)), mag["mz"].mean(-1)

    def at(arr, ref, off=0):
        idx = np.clip(np.array([int(np.argmin(np.abs(g - v))) for v in ref]) + off, 0, len(g) - 1)
        return arr[np.arange(len(ref)), idx]

    res["long_range_decomposition"] = {f"{name}{off:+d}": {"disconnected": float(np.mean(at(Dr, ref, off))),
                                                          "connected": float(np.mean(at(Kr, ref, off)))}
                                       for name, ref in (("entry", entry), ("exit", exit_)) for off in (-2, 0, 2)}
    res["magnetization_at_exit"] = {f"{off:+d}": {"staggered_abs": float(np.mean(at(ms, exit_, off))),
                                                  "uniform": float(np.mean(at(mu, exit_, off)))} for off in (-4, -2, 0, 2, 4)}
    X = (cache["gamma_pattern"][sel][:, None, :] * mag["mx"]).sum(-1) / L      # -dE/dGamma0 per site
    dX = np.diff(X, axis=1)
    mid = 0.5 * (g[1:] + g[:-1])
    typical = np.median(np.abs(dX), axis=1)

    def jump(ref):
        return np.array([dX[n, int(np.argmin(np.abs(mid - ref[n])))] for n in range(len(ref))]) / typical

    je, jx = jump(entry), jump(exit_)
    res["energy_derivative_jump_over_typical"] = {"entry_median": float(np.median(je)), "exit_median": float(np.median(jx)),
                                                  "entry_fraction_gt5": float(np.mean(np.abs(je) > 5)),
                                                  "exit_fraction_gt5": float(np.mean(np.abs(jx) > 5))}
    bd = np.argwhere(cache["branch_disagreement"][sel])
    res["branch_disagreements"] = {"points": int(len(bd)), "seeds": int(len(np.unique(bd[:, 0]))) if len(bd) else 0,
                                   "within_2_steps_of_entry": float(np.mean([abs(g[i] - entry[n]) <= 2 * step for n, i in bd])) if len(bd) else None,
                                   "within_2_steps_of_exit": float(np.mean([abs(g[i] - exit_[n]) <= 2 * step for n, i in bd])) if len(bd) else None}

    # 3. heterogeneity of the long-range pairs around the exit
    het = {}
    for d in (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15):
        fpos, cmax, cmean = [], [], []
        for n in range(sel.size):
            target = exit_[n] + d
            if target > g[-1] + 1e-9:
                continue
            c = np.diagonal(C[n, int(np.argmin(np.abs(g - target)))], offset=R)
            fpos.append(np.mean(c > 0))
            cmax.append(c.max())
            cmean.append(c.mean())
        het[f"{d:+.2f}"] = {"n": len(fpos), "mean_C": float(np.mean(cmean)), "fraction_pairs_positive": float(np.mean(fpos)),
                            "median_max_C": float(np.median(cmax))}
    res["exit_heterogeneity"] = het

    # 4. H0: frozen landscape velocity; entry peak, exit second wave, permutation null
    v = frozen["H0_velocity"][sel].astype(float)
    vmid = frozen["Gamma0_mid"]
    eligible = vmid - step / 2 >= step - 1e-12
    top_entry = np.array([vmid[eligible][np.argmax(vv[eligible])] for vv in v])
    res["H0_entry_global_max"] = localization(top_entry, entry, step)
    second, status = [], []
    for vv in v:
        w = vv.copy()
        w[~eligible] = -np.inf
        a = int(np.argmax(w))
        if a + 1 >= len(vmid) - 1:
            second.append(np.nan); status.append("short"); continue
        b = a + 1 + int(np.argmin(vv[a + 1:]))
        if b >= len(vmid) - 1:
            second.append(np.nan); status.append("no_second_wave"); continue
        c = b + int(np.argmax(vv[b:]))
        status.append("edge" if c == len(vmid) - 1 else "interior")
        second.append(vmid[c])
    second, status = np.asarray(second), np.asarray(status)
    res["H0_second_wave_status"] = {s: int((status == s).sum()) for s in np.unique(status)}
    interior = status == "interior"
    sharp = slopes_x > np.median(slopes_x)
    res["H0_second_wave_vs_exit"] = {
        "interior_all": localization(np.where(interior, second, np.nan), exit_, step),
        "interior_old": localization(np.where(interior & old, second, np.nan), exit_, step),
        "interior_new": localization(np.where(interior & ~old, second, np.nan), exit_, step),
        "interior_sharper_half": localization(np.where(interior & sharp, second, np.nan), exit_, step),
        "interior_smoother_half": localization(np.where(interior & ~sharp, second, np.nan), exit_, step)}
    vlog = np.log(v / np.median(v, axis=1, keepdims=True))
    obs = aligned_bump(np.exp(vlog), vmid, exit_, step, after=entry)
    null = np.array([aligned_bump(np.exp(vlog), vmid, exit_[RNG.permutation(sel.size)], step, after=entry) for _ in range(N_PERM)])
    res["H0_exit_aligned_bump"] = {"log_ratio": obs, "null_mean": float(null.mean()), "null_q95": float(np.percentile(null, 95)),
                                   "perm_p": float((np.sum(null >= obs) + 1) / (N_PERM + 1)),
                                   "entry_log_ratio": aligned_bump(np.exp(vlog), vmid, entry, step)}
    # error versus intrinsic width (sharpness law) and physical r +/- 2 ambiguity
    law = []
    for n in range(sel.size):
        cv = pair_curve(C[n], R)
        amb_e, amb_x = [], []
        for r in (R - 2, R + 2):
            (e2, _), (x2, _) = entry_exit(g, pair_curve(C[n], r))
            amb_e.append(abs(e2 - entry[n]))
            amb_x.append(abs(x2 - exit_[n]))
        law.append(("entry", amp[n] / slopes_e[n], abs(top_entry[n] - entry[n]), np.nanmean(amb_e) if np.isfinite(amb_e).any() else np.nan))
        law.append(("exit", amp[n] / slopes_x[n], abs(second[n] - exit_[n]) if interior[n] else np.nan, np.nanmean(amb_x) if np.isfinite(amb_x).any() else np.nan))
    res["sharpness_law_medians"] = {k: {"width": float(np.median([r[1] for r in law if r[0] == k])),
                                        "topological_error": float(np.nanmedian([r[2] for r in law if r[0] == k])),
                                        "physical_r_pm2_ambiguity": float(np.nanmedian([r[3] for r in law if r[0] == k]))}
                                    for k in ("entry", "exit")}

    # 4b. physical bulk response as a localizer (same peak rules as H0)
    a_, b_ = L // 4, 3 * L // 4
    resp = -np.gradient(C[:, :, a_:b_, a_:b_].mean((2, 3)), g, axis=1)
    phys_entry = np.array([g[1:][np.argmax(rr[1:])] for rr in resp])
    phys_exit = np.array([g[g > e + 2 * step][np.argmax(rr[g > e + 2 * step])] for rr, e in zip(resp, entry)])
    res["physical_response_localization"] = {"global_max_vs_entry": localization(phys_entry, entry, step),
                                             "post_entry_max_vs_exit": localization(phys_exit, exit_, step)}

    # 5. H0 of disconnected (magnetization profile) vs connected parts
    idx = np.arange(L)
    dis_m = dis.copy(); dis_m[..., idx, idx] = 1.0
    con_m = C - dis; con_m[..., idx, idx] = 1.0
    parts = {}
    for name, stack in (("full", C), ("disconnected", dis_m), ("connected", con_m)):
        vel = np.array([landscape_velocity(g, stack[n]) for n in range(sel.size)])
        pk = np.array([g[1:][np.argmax(vv[1:])] for vv in vel])
        parts[name] = {"entry_bump": aligned_bump(vel, g, entry, step), "exit_bump": aligned_bump(vel, g, exit_, step, after=entry),
                       "global_max_vs_entry_MAE": float(np.mean(np.abs(pk - entry)))}
    res["H0_by_part"] = parts

    # 6. H1 local motifs: location on weak bonds, timing across the exit
    gaps = np.stack([np.stack([square_gaps(C[n, i]) for i in range(len(g))]) for n in range(sel.size)])
    active = gaps > H1_TAU
    fr = {}
    for d in (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1):
        f = [active[n, int(np.argmin(np.abs(g - (exit_[n] + d))))].any() for n in range(sel.size)
             if abs(g[int(np.argmin(np.abs(g - (exit_[n] + d))))] - (exit_[n] + d)) <= step]
        fr[f"{d:+.2f}"] = float(np.mean(f))
    J = cache["J"][sel]
    Jsq = np.concatenate([J[n, 1:L - 2][active[n, -1]] for n in range(sel.size)])
    Jall = J[:, 1:L - 2].ravel()
    th = frozen["thresholds"]
    ti = int(np.argmin(np.abs(th - H1_TAU)))
    h1 = frozen["H1_cycle_count"][sel][:, :, ti]
    fr_real = {}
    for d in (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1):
        f = [h1[n, int(np.argmin(np.abs(g - (exit_[n] + d))))] > 0 for n in range(sel.size)
             if abs(g[int(np.argmin(np.abs(g - (exit_[n] + d))))] - (exit_[n] + d)) <= step]
        fr_real[f"{d:+.2f}"] = float(np.mean(f))
    res["H1_frozen_counts"] = {"tau": float(th[ti]), "fraction_seeds_with_H1_vs_exit_offset": fr_real}
    with (OUT / f"per_seed_L{L}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "cohort", "entry", "exit", "entry_slope", "exit_slope", "entry_width", "exit_width",
                         "H0_entry_peak", "H0_second_wave", "second_wave_status", "dEdG_jump_entry", "dEdG_jump_exit"])
        for n, k in enumerate(sel):
            writer.writerow([int(cache["seeds"][k]), cache["cohort"][k], entry[n], exit_[n], slopes_e[n], slopes_x[n],
                             amp[n] / slopes_e[n], amp[n] / slopes_x[n], top_entry[n], second[n], status[n], je[n], jx[n]])
    res["H1_square_motif"] = {"tau": H1_TAU, "fraction_seeds_active_vs_exit_offset": fr,
                              "middle_bond_J_median_at_motifs": float(np.median(Jsq)), "middle_bond_J_median_all": float(np.median(Jall)),
                              "mann_whitney_p": float(mannwhitneyu(Jsq, Jall).pvalue), "n_motifs_last_field": int(Jsq.size)}
    curves = {"g": g, "vmid": vmid, "v": v, "entry": entry, "exit": exit_, "ms": ms, "mu": mu, "X": X, "C": C, "R": R,
              "law": law, "active": active, "Jsq": Jsq, "Jall": Jall}
    return res, curves


def figure(curves):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for row, (L, c) in enumerate(curves.items()):
        g, step = c["g"], c["g"][1] - c["g"][0]
        bins = np.arange(-0.3, 0.3001, step)
        centres = 0.5 * (bins[1:] + bins[:-1])

        def aligned(y, x, ref):
            out = []
            for b in range(len(bins) - 1):
                vals = [y[n][(x - ref[n] >= bins[b]) & (x - ref[n] < bins[b + 1])] for n in range(len(ref))]
                vals = np.concatenate(vals)
                out.append(vals.mean() if vals.size else np.nan)
            return np.array(out)

        ax = axes[row, 0]
        phys = np.array([pair_curve(c["C"][n], c["R"]) for n in range(len(c["entry"]))])
        for ref, col, lab in ((c["entry"], "#0072B2", "aligned at entry"), (c["exit"], "#D55E00", "aligned at exit")):
            ax.plot(centres, aligned(phys, g, ref), color=col, label=lab)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set(title=f"L={L}: signed correlator at r=L/2-1", xlabel=r"$\Gamma_0-\Gamma_0^{\rm cross}$")
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        vlog = np.log(c["v"] / np.median(c["v"], axis=1, keepdims=True))
        for ref, col, lab in ((c["entry"], "#0072B2", "entry"), (c["exit"], "#D55E00", "exit")):
            ax.plot(centres, aligned(vlog, c["vmid"], ref), color=col, label=f"aligned at {lab}")
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set(title=r"H$_0$ landscape velocity (log, per-seed normalized)", xlabel=r"$\Gamma_0-\Gamma_0^{\rm cross}$")
        ax.legend(fontsize=8)

        ax = axes[row, 2]
        ax.plot(centres, aligned(c["ms"], g, c["exit"]), color="#0072B2", label=r"$|m_s|$ staggered")
        ax.plot(centres, aligned(c["mu"], g, c["exit"]), color="#D55E00", label=r"$m_u$ uniform")
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set(title="magnetization, aligned at exit", xlabel=r"$\Gamma_0-\Gamma_0^{\rm exit}$")
        ax.legend(fontsize=8)

        ax = axes[row, 3]
        law = c["law"]
        for kind, col in (("entry", "#0072B2"), ("exit", "#D55E00")):
            w = np.array([r[1] for r in law if r[0] == kind])
            e = np.array([r[2] for r in law if r[0] == kind])
            a = np.array([r[3] for r in law if r[0] == kind])
            ax.scatter(w, e + 1e-3, s=10, color=col, alpha=0.6, label=f"{kind}: topology")
            ax.scatter(w, a + 1e-3, s=10, marker="x", color=col, alpha=0.4, label=f"{kind}: physical r$\\pm$2")
        ax.plot([3e-3, 2], [3e-3, 2], "k:", lw=0.8)
        ax.set(xscale="log", yscale="log", title="error vs intrinsic crossing width", xlabel="width = A / |dC/d$\\Gamma_0$|")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "reentrant_exit_anatomy_overview.png", dpi=110)
    fig.savefig(OUT / "reentrant_exit_anatomy_overview.pdf")
    plt.close(fig)


def main():
    summary, curves = {}, {}
    for L in (16, 32):
        res, cur = analyse(L)
        summary[f"L{L}"] = res
        curves[L] = cur
        print(json.dumps({k: res[k] for k in ("crossing_width_median", "energy_derivative_jump_over_typical",
                                                 "H0_exit_aligned_bump", "H1_square_motif")}, indent=1), flush=True)
    summary["notes"] = {
        "selection": "frozen physical selection of the all-1000 analyses (97 L16, 137 L32 reentrant seeds)",
        "H0_velocity": "frozen H0 landscape velocity arrays; second wave = max after the first minimum following the global maximum",
        "magnetization": "measured on saved MPS (no DMRG); Czz validated against stored values to ~1e-15",
        "H1_proxy": f"four-site square motif gap > {H1_TAU}",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure(curves)


if __name__ == "__main__":
    main()
