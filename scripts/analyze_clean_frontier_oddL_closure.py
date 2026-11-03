"""Clean OBC odd-L frontier closure: apply the rules frozen on 2026-10-15.

Rules: documentation/PLAN_CLEAN_FRONTIER_ODDL_CLOSURE_20261015.md (written before the
new cuts were simulated). L63 and L31, h = 1.1-1.9, full Gamma 0-1.2, chi = 128.
C = Cstag raw, D = clip(1 - C, 0, 2) symmetrized, Vietoris-Rips with gudhi, Z2, dim <= 2.

Primary H1 estimator: bulk bars (end-closure ring removed: death triangle contains
site 0 or L-1, or birth edge {L//2-1, L//2+1}); N1(tau) = # bulk bars with lifetime
> tau; estimate = midpoint of the QC-valid interval with the largest positive
dN1/dGamma (first on ties); none -> "no H1". tau = 0.006 primary, 0.008 alternative.
Robustness: same on D restricted to sites k..L-1-k (k = 1, 2, 3), all bars counted.
Secondary: H0 closed-form landscape-velocity peak with QC mask, half-chain entropy peak.
Amendment v2 (2026-10-15 12:30 UTC, main estimator): all bars including the ring,
estimate = midpoint of the first QC-valid interval where N1(tau=0.04) rises; tau 0.03
and 0.06 as sensitivity; train/test split reported separately.
Decisions: hit if |estimate - digitized frontier| <= 0.05; h* = smallest h of the grid
where L63 primary hits at that h and at every larger h; size trend claimed only if
err(L63) < err(L31) in the majority of cuts with h >= h*.

Only cuts whose campaign is complete are used. While cuts are missing every decision
is reported as provisional. Read-only on results; per-scan H1 counts are cached.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from build_clean_frontier_H0_H1_L64_expanded import analyze_curve

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "clean_ising_gamma_campaign"
OUTPUT = BASE / "clean-frontier-oddL-closure-v1" / "analysis"
CACHE = OUTPUT / "cache"
PLAN = "documentation/PLAN_CLEAN_FRONTIER_ODDL_CLOSURE_20261015.md"
SIZES = (31, 63)
H_GRID = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9)
TAUS = (0.006, 0.008)
TAU_PRIMARY = 0.006
TRIMS = (1, 2, 3)
HIT = 0.05
TRIM_DISCREPANCY = 0.03
CHI = 128
# Digitized Lajko et al. (2020) Fig. 1, linear interpolation (frozen in the plan).
FRONTIER = {1.1: 0.757, 1.2: 0.712, 1.3: 0.656, 1.4: 0.599, 1.5: 0.526,
            1.6: 0.453, 1.7: 0.366, 1.8: 0.278, 1.9: 0.154}
FIRST_CACHE_VERSION = 1
# Amendment v2 (frozen 2026-10-15 12:30 UTC, see PLAN): all bars, first rise of N1, jump >= 1.
V2_TAUS = (0.04, 0.03, 0.06)
V2_PRIMARY = "v2_tau0.04"
TRAIN = {(31, 1.1), (31, 1.2), (31, 1.3), (31, 1.4), (63, 1.3), (63, 1.7)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def campaign(L: int, h: float) -> Path:
    tag = "h" + f"{h:.1f}".replace(".", "")
    return BASE / f"clean-L{L}-oddL-fullgamma-{tag}-c128-v1"


def complete_scan(L: int, h: float) -> Path | None:
    camp = campaign(L, h)
    status = camp / "campaign_status.json"
    if not status.exists():
        return None
    jobs = json.loads(status.read_text(encoding="utf-8"))["jobs"]
    if not jobs or any(job.get("status") != "complete" for job in jobs):
        return None
    path = camp / "scans" / f"L{L}_h_{h:g}_J1_chi{CHI}" / "continuation_summary.npz"
    return path if path.exists() else None


def distance(cstag: np.ndarray) -> np.ndarray:
    d = np.clip(1.0 - 0.5 * (cstag + cstag.T), 0.0, 2.0)
    np.fill_diagonal(d, 0.0)
    return d


def h1_bars(dist: np.ndarray, ring_rule: bool) -> list[tuple[float, bool]]:
    """(lifetime, is_end_closure_ring) for every finite H1 bar."""
    import gudhi

    n = dist.shape[0]
    centre = {n // 2 - 1, n // 2 + 1}
    tree = gudhi.RipsComplex(distance_matrix=dist, max_edge_length=2.0).create_simplex_tree(max_dimension=2)
    tree.persistence(homology_coeff_field=2, min_persistence=-1.0)
    bars = []
    for birth, death in tree.persistence_pairs():
        if len(birth) != 2 or not death:
            continue
        life = tree.filtration(death) - tree.filtration(birth)
        if life > 1e-14:
            ring = ring_rule and (bool({0, n - 1} & set(death)) or set(birth) == centre)
            bars.append((float(life), ring))
    return bars


def h1_counts(cstag: np.ndarray) -> dict[str, np.ndarray]:
    """N1(tau) per Gamma for the primary rule and for each trimmed chain."""
    out = {}
    G, L = cstag.shape[0], cstag.shape[1]
    for tau in TAUS:
        out[f"primary_tau{tau}"] = np.zeros(G, int)
        out[f"ring_tau{tau}"] = np.zeros(G, int)
        for k in TRIMS:
            out[f"trim{k}_tau{tau}"] = np.zeros(G, int)
    for g in range(G):
        bars = h1_bars(distance(cstag[g]), ring_rule=True)
        for tau in TAUS:
            out[f"primary_tau{tau}"][g] = sum(life > tau and not ring for life, ring in bars)
            out[f"ring_tau{tau}"][g] = sum(life > tau and ring for life, ring in bars)
        for k in TRIMS:
            sub = cstag[g][k:L - k, k:L - k]
            trimmed = h1_bars(distance(sub), ring_rule=False)
            for tau in TAUS:
                out[f"trim{k}_tau{tau}"][g] = sum(life > tau for life, _ in trimmed)
    return out


def cached_counts(path: Path, cstag: np.ndarray) -> dict[str, np.ndarray]:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = sha256(path)[:16]
    cache = CACHE / f"{path.parent.name}_{key}_v{FIRST_CACHE_VERSION}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return {k: np.asarray(z[k]) for k in z.files}
    counts = h1_counts(cstag)
    np.savez_compressed(cache, **counts)
    return counts


def proliferation(gamma: np.ndarray, n1: np.ndarray, valid_interval: np.ndarray) -> float | None:
    """Midpoint of the valid interval with the largest positive dN1/dGamma (first on ties)."""
    # Round the grid steps so that equal jumps on a uniform grid tie exactly (frozen rule: first wins).
    slope = np.round(np.diff(n1) / np.round(np.diff(gamma), 10), 6)
    slope = np.where(valid_interval, slope, -np.inf)
    if not np.any(slope > 0):
        return None
    i = int(np.argmax(slope))  # argmax returns the first maximum
    return float(0.5 * (gamma[i] + gamma[i + 1]))


def analyze_cut(L: int, h: float, path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as z:
        gamma = np.asarray(z["Gamma_values"], float)
        cstag = np.asarray(z["Cstag"], float)
        bond = np.asarray(z["selected_max_bond_dimension"], int)
        disagree = np.asarray(z["branch_disagreement"], bool)
        entropy = np.asarray(z["half_chain_entropy"], float)
    if cstag.shape[1] != L:
        raise ValueError(f"{path}: expected L={L}, found {cstag.shape[1]}")
    invalid = disagree | (bond >= CHI)
    valid_interval = ~(invalid[:-1] | invalid[1:])
    counts = cached_counts(path, cstag)
    h0 = analyze_curve({"Gamma": gamma, "Cstag": cstag, "bond": bond, "branch_disagreement": disagree})
    frontier = FRONTIER[h]
    est = {}
    for name, n1 in counts.items():
        if name.startswith("ring"):
            continue
        value = proliferation(gamma, n1, valid_interval)
        est[name] = {"estimate": value,
                     "error": None if value is None else abs(value - frontier),
                     "hit": None if value is None else abs(value - frontier) <= HIT}
    from explore_clean_frontier_oddL_h1_params import bars_for, estimate as first_rise
    bars = bars_for(path)
    for tau in V2_TAUS:
        value = first_rise(bars, tau, 1, "all", "first")
        est[f"v2_tau{tau}"] = {"estimate": value,
                               "error": None if value is None else abs(value - frontier),
                               "hit": None if value is None else abs(value - frontier) <= HIT}
    flags = {}
    for tau in TAUS:
        p = est[f"primary_tau{tau}"]["estimate"]
        for k in TRIMS:
            t = est[f"trim{k}_tau{tau}"]["estimate"]
            flags[f"trim{k}_tau{tau}_discrepant"] = (
                (p is None) != (t is None) or (p is not None and abs(p - t) > TRIM_DISCREPANCY))
    row = {
        "L": L, "h": h, "source": str(path.relative_to(ROOT)), "sha256": sha256(path),
        "frontier": frontier, "qc_invalid_points": int(invalid.sum()),
        "H0_peak": float(h0["H0_closed_peak"]), "H0_error": abs(float(h0["H0_closed_peak"]) - frontier),
        "entropy_peak": float(gamma[int(np.argmax(entropy))]),
        "estimates": est, "trim_flags": flags,
        "ring_present_tau0.006": bool(np.any(counts["ring_tau0.006"] > 0)),
    }
    per_gamma = {"Gamma": gamma, "qc_invalid": invalid, **counts}
    return row, per_gamma


def h_star(rows: dict[tuple[int, float], dict], name: str) -> dict:
    """Smallest h with an L63 primary hit there and at every larger h of the grid."""
    missing = [h for h in H_GRID if (63, h) not in rows]
    hits = {h: rows[(63, h)]["estimates"][name]["hit"] for h in H_GRID if (63, h) in rows}
    star = None
    for h in sorted(hits, reverse=True):
        if hits[h]:
            star = h
        else:
            break
    if star is not None and missing and any(m > star for m in missing):
        star_status = "provisional: missing L63 cuts above h*"
    elif missing:
        star_status = "provisional: missing L63 cuts below h* could still lower it"
    else:
        star_status = "final"
    if hits and max(hits) < H_GRID[-1]:
        star, star_status = None, "undefined: L63 h=1.9 not yet available"
    return {"estimator": name, "h_star": star, "status": star_status, "L63_missing": missing,
            "L63_hits": {f"{h:.1f}": v for h, v in hits.items()}}


def size_trend(rows: dict, star: float | None, name: str) -> dict:
    if star is None:
        return {"status": "not evaluated (no h*)"}
    better, compared = [], []
    for h in H_GRID:
        if h < star or (63, h) not in rows or (31, h) not in rows:
            continue
        e63 = rows[(63, h)]["estimates"][name]["error"]
        e31 = rows[(31, h)]["estimates"][name]["error"]
        if e63 is None or e31 is None:
            continue
        compared.append(h)
        better.append(e63 < e31)
    n = len(compared)
    return {"cuts_compared": compared, "L63_better": int(sum(better)),
            "claim_improves_with_L": bool(n and sum(better) > n / 2),
            "status": "final" if all((L, h) in rows for L in SIZES for h in H_GRID if h >= star) else "provisional"}


def plot(rows: dict) -> None:
    import matplotlib.pyplot as plt

    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    hs = np.array(H_GRID)
    ax.plot(hs, [FRONTIER[h] for h in H_GRID], "k-", lw=2, label="digitized frontier")
    style = {31: dict(color="#56B4E9", mfc="white"), 63: dict(color="#0072B2", mfc="#0072B2")}
    for L in SIZES:
        cut = [(h, rows[(L, h)]) for h in H_GRID if (L, h) in rows]
        if not cut:
            continue
        x = [h for h, _ in cut]
        ax.plot(x, [r["H0_peak"] for _, r in cut], "s--", color=style[L]["color"], mfc="white",
                label=f"L={L} H0 peak")
        y = [r["estimates"][f"primary_tau{TAU_PRIMARY}"]["estimate"] for _, r in cut]
        ax.plot([h for h, v in zip(x, y) if v is not None], [v for v in y if v is not None], "o",
                ms=5, alpha=0.5, label=f"L={L} H1 v1 bulk (tau={TAU_PRIMARY})", **style[L])
        y2 = [r["estimates"][V2_PRIMARY]["estimate"] for _, r in cut]
        ax.plot([h for h, v in zip(x, y2) if v is not None], [v for v in y2 if v is not None], "*",
                ms=11, color=style[L]["color"], label=f"L={L} H1 v2 first rise (tau=0.04)")
        err = [r["estimates"][V2_PRIMARY]["error"] for _, r in cut]
        ax_err.plot([h for h, e in zip(x, err) if e is not None], [e for e in err if e is not None],
                    "*-", ms=10, color=style[L]["color"], label=f"L={L} H1 v2")
        ax_err.plot(x, [r["H0_error"] for _, r in cut], "s--", color=style[L]["color"], mfc="white",
                    label=f"L={L} H0")
    ax_err.axhline(HIT, color="k", ls=":", lw=1, label=f"hit threshold {HIT}")
    ax.set(xlabel="h/J", ylabel=r"$\Gamma/J$", title="Clean OBC odd L: estimators vs frontier (exploratory)")
    ax_err.set(xlabel="h/J", ylabel="|estimate - frontier|")
    ax.legend(fontsize=6.5)
    ax_err.legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT / f"clean_frontier_oddL_closure.{ext}", dpi=160)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, per_gamma, missing = {}, {}, []
    for L in SIZES:
        for h in H_GRID:
            path = complete_scan(L, h)
            if path is None:
                missing.append(f"L{L} h={h}")
                continue
            print(f"analyzing L={L} h={h}", flush=True)
            rows[(L, h)], per_gamma[(L, h)] = analyze_cut(L, h, path)

    decisions = {}
    names = [f"v2_tau{t}" for t in V2_TAUS] + [f"primary_tau{t}" for t in TAUS]
    for name in names:
        star = h_star(rows, name)
        split = {}
        for label, keys in (("train", [k for k in rows if k in TRAIN]),
                            ("test", [k for k in rows if k not in TRAIN])):
            errs = [rows[k]["estimates"][name]["error"] for k in keys]
            got = [e for e in errs if e is not None]
            split[label] = {"cuts": [f"L{L}_h{h}" for L, h in sorted(keys)], "n_none": len(errs) - len(got),
                            "hits": int(sum(e <= HIT for e in got)),
                            "mae": float(np.mean(got)) if got else None}
        decisions[name] = {"h_star": star, "size_trend": size_trend(rows, star["h_star"], name), **split}
    discrepant = sorted({f"L{L} h={h} {k}" for (L, h), r in rows.items()
                         for k, v in r["trim_flags"].items() if v})

    with (OUTPUT / "per_cut.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["L", "h", "frontier", "H0_peak", "H0_error", "entropy_peak", "qc_invalid_points",
                  "ring_present_tau0.006"]
        names = sorted(next(iter(rows.values()))["estimates"]) if rows else []
        writer = csv.writer(handle)
        writer.writerow(fields + [f"{n}_estimate" for n in names] + [f"{n}_error" for n in names])
        for key in sorted(rows):
            r = rows[key]
            writer.writerow([r[f] for f in fields] + [r["estimates"][n]["estimate"] for n in names]
                            + [r["estimates"][n]["error"] for n in names])
    with (OUTPUT / "per_gamma.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        keys = None
        for (L, h), pg in sorted(per_gamma.items()):
            if keys is None:
                keys = [k for k in pg if k != "Gamma"]
                writer.writerow(["L", "h", "Gamma"] + keys)
            for i, g in enumerate(pg["Gamma"]):
                writer.writerow([L, h, round(float(g), 10)] + [int(pg[k][i]) for k in keys])

    summary = {
        "rules": PLAN,
        "status": "final" if not missing else "provisional",
        "missing_cuts": missing,
        "decisions": decisions,
        "trim_discrepancies_gt_0.03": discrepant,
        "cuts": {f"L{L}_h{h}": r for (L, h), r in sorted(rows.items())},
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(rows)
    print(json.dumps({"status": summary["status"], "missing": missing, "decisions": decisions,
                      "trim_discrepancies": discrepant}, indent=1))


if __name__ == "__main__":
    main()
