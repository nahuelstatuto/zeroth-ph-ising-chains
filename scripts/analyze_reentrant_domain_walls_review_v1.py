"""Anatomy of the reentrant entry from domain walls (review v1, referee 2 Sec. 4.4).

Question: are the entries with Delta M ~ 0 a reorganization of domain walls at constant
magnetization, as written in Sec. VI of v0, or the annihilation of a pair of walls that
bounds a reversed domain of even length (which leaves M unchanged)?

Per selected reentrant realization (L16, L32 even; L17, L33 odd), at the two grid points
that bracket its physical entry (ib < entry < ia):
  walls      nearest-neighbour pairs aligned along z, from the measured profiles
             (mz_i mz_{i+1} > 0 with |mz| >= MZ_MIN on both sites) and from the correlations
             (C_{i,i+1} < 0)
  flipped    sites whose staggered orientation s_i = (-1)^i sign(mz_i) changes between ib
             and ia (|mz| >= MZ_MIN on both points)
  blocks     maximal contiguous runs of flipped sites, their lengths and parities
  Delta M    sum_i mz_i(ia) - sum_i mz_i(ib) (as in v0)
Topological domains: classes of the H0 filtration at the natural scale D = 1 (components of
C_ij > 0, all distances) compared with the classes of s_i (adjusted Rand index), and
the number of class changes along the chain compared with the wall count.

Run (tn_tda not required):  python scripts/analyze_reentrant_domain_walls_review_v1.py
Output: results/review_v1/reentrant-domain-walls-v1/summary.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_sdrg_h0_clusters_review_v1 import ari, components  # noqa: E402

BASE = ROOT / "results" / "random_ising_gamma_ensemble"
EVEN = BASE / "reentrant-exit-anatomy-v1" / "analysis"
ODD = BASE / "reentrant-oddL-L17-L33-v1" / "analysis"
OUT = ROOT / "results" / "review_v1" / "reentrant-domain-walls-v1"
MZ_MIN = 0.2
ODD_PREFIX = {17: "L17-reentrant-h2500", 33: "L33-reentrant-h23125"}


def scan_dirs(prefix):
    rows = []
    for st in sorted(BASE.glob(f"{prefix}-oddL-*/campaign_status.json")):
        for j in json.loads(st.read_text(encoding="utf-8"))["jobs"]:
            rows.append((int(j["seed"]), Path(j["target_directory"])))
    return dict(rows)


def load(L):
    if L % 2 == 0:
        cache = np.load(EVEN / f"cache_L{L}.npz")
        mag = np.load(EVEN / f"local_magnetization_L{L}.npz")
        sel = mag["sel"]
        return cache["Gamma0"], cache["Cstag"][sel].astype(float), mag["mz"], cache["entry"][sel], cache["exit"][sel]
    cache = np.load(ODD / f"cache_L{L}.npz")
    mag = np.load(ODD / f"local_magnetization_L{L}.npz")
    dirs = scan_dirs(ODD_PREFIX[L])
    C = np.stack([np.load(dirs[int(s)] / "continuation_summary.npz")["Cstag"] for s in mag["seeds"]]).astype(float)
    return cache["Gamma0"], C, mag["mz"], cache["entry"], cache["exit"]


def walls_mz(m):
    ok = (np.abs(m[:-1]) >= MZ_MIN) & (np.abs(m[1:]) >= MZ_MIN)
    return np.flatnonzero(ok & (m[:-1] * m[1:] > 0))


def stag(m):
    return (-1) ** np.arange(m.size) * np.sign(m)


def blocks(mask):
    runs, start = [], None
    for i, v in enumerate(np.r_[mask, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    return runs


def analyse(L):
    g, C, mz, entry, exit_ = load(L)
    events, aris, wall_match = [], [], []
    for n in range(mz.shape[0]):
        ib = int(np.searchsorted(g, entry[n]) - 1)
        ia = ib + 1
        mb, ma = mz[n, ib], mz[n, ia]
        firm = (np.abs(mb) >= MZ_MIN) & (np.abs(ma) >= MZ_MIN)
        flipped = firm & (stag(mb) != stag(ma))
        runs = blocks(flipped)
        wb, wa = walls_mz(mb), walls_mz(ma)
        cb = int(np.sum(np.diagonal(C[n, ib], 1) < 0))
        ca = int(np.sum(np.diagonal(C[n, ia], 1) < 0))
        dM = float(ma.sum() - mb.sum())
        interior = [r for r in runs if r[0] > 0 and r[1] < L - 1]
        events.append({"dM": dM, "walls_before": int(wb.size), "walls_after": int(wa.size), "dW": int(wa.size - wb.size),
                       "walls_C_before": cb, "walls_C_after": ca, "n_blocks": len(runs),
                       "block_lengths": [r[1] - r[0] + 1 for r in runs], "interior_blocks": len(interior),
                       "touches_end": any(r[0] == 0 or r[1] == L - 1 for r in runs), "firm_fraction": float(firm.mean())})
        for k in (ib, ia):
            s = stag(mz[n, k])
            good = np.abs(mz[n, k]) >= MZ_MIN
            if good.sum() >= 4:
                h0 = components(L, [(i, j) for i in range(L) for j in range(i + 1, L) if C[n, k, i, j] > 0])
                aris.append(ari(s[good], h0[good]))
                changes_h0 = int(np.sum(h0[:-1] != h0[1:]))
                wall_match.append(changes_h0 == int(np.sum(np.diagonal(C[n, k], 1) < 0)))
    dM = np.array([e["dM"] for e in events])
    cls = {"dM~0": np.abs(dM) < 0.5, "dM~-2": np.abs(dM + 2) < 0.5}
    out = {"realizations": len(events), "H0_classes_vs_staggered_sign_ARI_mean": float(np.mean(aris)),
           "H0_class_changes_equal_C_walls_fraction": float(np.mean(wall_match))}
    for name, mask in cls.items():
        ev = [e for e, m in zip(events, mask) if m]
        if not ev:
            continue
        single = [e for e in ev if e["n_blocks"] == 1]
        out[name] = {
            "count": len(ev),
            "dW_hist": dict(sorted(Counter(e["dW"] for e in ev).items())),
            "dW_C_hist": dict(sorted(Counter(e["walls_C_after"] - e["walls_C_before"] for e in ev).items())),
            "n_blocks_hist": dict(sorted(Counter(e["n_blocks"] for e in ev).items())),
            "single_block_fraction": len(single) / len(ev),
            "single_block_length_parity_even_fraction": float(np.mean([e["block_lengths"][0] % 2 == 0 for e in single])) if single else None,
            "single_block_interior_fraction": float(np.mean([e["interior_blocks"] == 1 for e in single])) if single else None,
            "touches_end_fraction": float(np.mean([e["touches_end"] for e in ev])),
        }
    out["events"] = events
    print(f"L{L}: {len(events)} entries | ARI(H0 classes, staggered sign) {out['H0_classes_vs_staggered_sign_ARI_mean']:.3f}, "
          f"class changes = C walls in {out['H0_class_changes_equal_C_walls_fraction']:.3f}", flush=True)
    for name in cls:
        if name in out:
            o = out[name]
            print(f"   {name}: n={o['count']} dW {o['dW_hist']} dW_C {o['dW_C_hist']} blocks {o['n_blocks_hist']} "
                  f"single-block even {o['single_block_length_parity_even_fraction']} interior {o['single_block_interior_fraction']} "
                  f"touches end {o['touches_end_fraction']:.2f}", flush=True)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {f"L{L}": analyse(L) for L in (16, 17, 32, 33)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
