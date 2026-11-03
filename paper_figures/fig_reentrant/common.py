"""Shared loading and estimators for the reentrant figures R1-R3."""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_common"))
import paper_style as ps  # noqa: E402

DATA_DIR = HERE / "data_raw"
OUTPUT_DIR = HERE / "output"
SIZES = (16, 32)
LS = {16: "--", 32: "-"}           # line style per size (colour also follows size)
ALIGN_RANGE = 0.3


def lettered_titles(axes):
    """Bold '(a)' panel letters placed as in the other figures (ps.panel_labels); titles stay left-aligned."""
    for ax in axes:
        title = ax.get_title()
        ax.set_title("")
        if title:
            ax.set_title(title, loc="left", fontsize=10)
    ps.panel_labels(axes, "abcdefghij")


def load():
    ps.verify_manifest(DATA_DIR)
    return {L: dict(np.load(DATA_DIR / f"reentrant_L{L}.npz", allow_pickle=False)) for L in SIZES}


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


def aligned(values, x, ref, step, statistic="mean"):
    """Bin values[n, :] at x - ref[n]; returns centres, statistic and per-bin samples."""
    bins = np.arange(-ALIGN_RANGE - step / 2, ALIGN_RANGE + step, step)
    centres = 0.5 * (bins[1:] + bins[:-1])
    dx = x[None, :] - ref[:, None]
    samples = [values[(dx >= lo) & (dx < hi)] for lo, hi in zip(bins[:-1], bins[1:])]
    func = {"mean": np.mean, "median": np.median}[statistic]
    stat = np.array([func(s) if s.size else np.nan for s in samples])
    return centres, stat, samples


def log_normalized_velocity(d):
    v = d["H0_velocity"].astype(float)
    return np.log(v / np.median(v, axis=1, keepdims=True))


def topological_entry(d):
    v, mid = d["H0_velocity"].astype(float), d["Gamma0_mid"]
    step = d["Gamma0"][1] - d["Gamma0"][0]
    eligible = mid - step / 2 >= step - 1e-12          # frozen rule: first interval excluded
    return np.array([mid[eligible][np.argmax(vv[eligible])] for vv in v])


def second_wave(d):
    """Maximum of the H0 velocity after the first minimum that follows its global maximum."""
    v, mid = d["H0_velocity"].astype(float), d["Gamma0_mid"]
    step = d["Gamma0"][1] - d["Gamma0"][0]
    eligible = mid - step / 2 >= step - 1e-12
    est, status = [], []
    for vv in v:
        w = np.where(eligible, vv, -np.inf)
        a = int(np.argmax(w))
        if a + 1 >= len(mid) - 1:
            est.append(np.nan); status.append("none"); continue
        b = a + 1 + int(np.argmin(vv[a + 1:]))
        if b >= len(mid) - 1:
            est.append(np.nan); status.append("none"); continue
        c = b + int(np.argmax(vv[b:]))
        est.append(mid[c]); status.append("edge" if c == len(mid) - 1 else "interior")
    return np.array(est), np.array(status)


def crossing_geometry(d):
    """Entry/exit slope, lobe amplitude and the r +/- 2 physical ambiguity, per seed."""
    g, R = d["Gamma0"], int(d["R"])
    odd_r = list(d["odd_r"])
    iR = odd_r.index(R)
    rows = []
    for n in range(len(d["entry"])):
        cv = d["curves_r"][n, iR]
        (e, se), (x, sx) = entry_exit(g, cv)
        amb_e, amb_x = [], []
        for r in (R - 2, R + 2):
            if r in odd_r:
                (e2, _), (x2, _) = entry_exit(g, d["curves_r"][n, odd_r.index(r)])
                amb_e.append(abs(e2 - e))
                amb_x.append(abs(x2 - x))
        rows.append({"amp": cv.max(), "slope_entry": se, "slope_exit": abs(sx),
                     "amb_entry": np.nanmean(amb_e) if np.isfinite(amb_e).any() else np.nan,
                     "amb_exit": np.nanmean(amb_x) if np.isfinite(amb_x).any() else np.nan})
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def split_velocity(d, w):
    """Long-range landscape (frozen L16 rule): entry/exit = velocity maxima left/right of the apex of M_w."""
    g, mid = d["Gamma0"], d["Gamma0_mid"]
    step = g[1] - g[0]
    first = 2 * step
    en, ex = [], []
    for v, m in zip(d[f"LR_velocity_w{w}"], d[f"LR_level_w{w}"]):
        el = np.flatnonzero(g >= first - 1e-12)
        apex = g[el[np.argmax(m[el])]]
        left = np.flatnonzero((mid >= first + step / 2 - 1e-12) & (mid <= apex + 1e-12))
        right = np.flatnonzero(mid > apex + 1e-12)
        en.append(mid[left[np.argmax(v[left])]] if left.size else np.nan)
        ex.append(mid[right[np.argmax(v[right])]] if right.size else np.nan)
    return np.array(en), np.array(ex)


def level_crossings(d, w):
    """Entry/exit = zero crossings of the long-range level M_w (w=0: identical to C(L/2-1))."""
    g = d["Gamma0"]
    return np.array([[c[0] for c in entry_exit(g, m)] for m in d[f"LR_level_w{w}"]]).T
