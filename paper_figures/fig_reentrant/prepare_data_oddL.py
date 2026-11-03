"""Add the odd-L reentrant cohorts (OBC L17 h=2.5, L33 h=2.3125; 400 seeds each) to the reentrant figures.

Selection, entry and exit: frozen physical selection applied by
scripts/analyze_reentrant_oddL_L17_L33.py (cache_L{L}.npz of reentrant-oddL-L17-L33-v1),
R = floor(L/2) - 1. For the selected seeds this file stores the quantities the main
figure needs, computed with the same definitions as the even cohorts:
  H0 landscape velocity (all pairs, frozen GRID/LAYERS, forward differences),
  long-range landscape levels and velocities w = 0, 1, 2 (prepare_data.long_range),
  H1 interval counts with lifetime > 0.006 (Vietoris-Rips on clip(1-C,0,2), Z2),
  staggered and uniform magnetizations, -dE/dGamma0 and long-range pair statistics for
  Fig. S1, from local_magnetization_L{L}.npz (scripts/measure_reentrant_oddL_local_magnetization.py).
Output: data_raw/reentrant_L{17,33}.npz; source_manifest.json is extended, not rewritten.
Run from the project root:  python paper_figures/fig_reentrant/prepare_data_oddL.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_data import ROOT, OUT, BASE, H1_TAU, sha256, long_range  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from analyze_clean_L64_new_topology import h1_diagram, mst_deaths  # noqa: E402
from analyze_obc_L32_existing_H0_H1_replication import GRID, LAYERS, landscape_from_deaths  # noqa: E402

ODD = BASE / "reentrant-oddL-L17-L33-v1" / "analysis"
COHORTS = {17: ("L17-reentrant-h2500", 2.5), 33: ("L33-reentrant-h23125", 2.3125)}


def scan_dirs(prefix):
    rows = []
    for st in sorted((BASE).glob(f"{prefix}-oddL-*/campaign_status.json")):
        for j in json.loads(st.read_text(encoding="utf-8"))["jobs"]:
            rows.append((int(j["seed"]), Path(j["target_directory"])))
    return dict(sorted(rows))


def h0_velocity(C, g):
    lands = np.stack([landscape_from_deaths(mst_deaths(C[i])) for i in range(g.size)]).astype(float)
    d = np.diff(lands, axis=0) / np.diff(g)[:, None, None]
    return np.sqrt(np.trapezoid(np.sum(d ** 2, axis=1), GRID, axis=1) / LAYERS)


def main():
    sources, files = [], []
    for L, (prefix, h) in COHORTS.items():
        cache_path = ODD / f"cache_L{L}.npz"
        cache = np.load(cache_path)
        dirs = scan_dirs(prefix)
        seeds = cache["seeds"][cache["selected"]]
        g = np.asarray(cache["Gamma0"], float)
        C = []
        for s in seeds:
            path = dirs[int(s)] / "continuation_summary.npz"
            sources.append(path)
            with np.load(path) as z:
                if not np.allclose(z["Gamma0_values"], g):
                    raise ValueError(f"grid mismatch {path}")
                C.append(np.asarray(z["Cstag"], float))
        C = np.stack(C)
        R = L // 2 - 1
        odd_r = np.arange(3, L - 2, 2)
        curves_r = np.stack([np.diagonal(C, offset=r, axis1=-2, axis2=-1).mean(-1) for r in odd_r], axis=1)
        if not np.allclose(curves_r[:, list(odd_r).index(R)], cache["curves_R"][cache["selected"]]):
            raise ValueError("C(R) mismatch with the selection cache")
        mag_path = ODD / f"local_magnetization_L{L}.npz"
        mag = np.load(mag_path)
        if not np.array_equal(mag["seeds"], seeds):
            raise ValueError("magnetization seeds mismatch")
        sig = mag["mz"] * ((-1) ** np.arange(L))[None, None, :]
        pairs = np.diagonal(C, offset=R, axis1=-2, axis2=-1)
        sources.append(mag_path)
        h1 = np.array([[np.sum(np.diff(h1_diagram(C[n, i]), axis=1) > H1_TAU) for i in range(g.size)]
                       for n in range(len(seeds))])
        target = OUT / f"reentrant_L{L}.npz"
        np.savez_compressed(
            target, L=L, h=h, seeds=seeds, Gamma0=g, Gamma0_mid=0.5 * (g[1:] + g[:-1]),
            entry=cache["entry"], exit=cache["exit"], odd_r=odd_r, curves_r=curves_r.astype(np.float32), R=R,
            H0_velocity=np.stack([h0_velocity(C[n], g) for n in range(len(seeds))]),
            H1_count=h1, H1_tau=H1_TAU, staggered_abs=np.abs(sig.mean(-1)), uniform=mag["mz"].mean(-1),
            minus_dE_dGamma0=(mag["gamma_pattern"][:, None, :] * mag["mx"]).sum(-1) / L,
            pair_fraction_positive=(pairs > 0).mean(-1), pair_max=pairs.max(-1), **long_range(C, g, L))
        sources.append(cache_path)
        files.append(target)
        print(L, len(seeds), "seeds", flush=True)
    manifest_path = OUT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = {f.name for f in files}
    manifest["files"] = [f for f in manifest["files"] if f["file"] not in names]
    manifest["files"] += [{"file": f.name, "sha256": sha256(f), "size_bytes": f.stat().st_size} for f in files]
    known = {s["path"] for s in manifest["sources"]}
    manifest["sources"] += [{"path": str(Path(s).relative_to(ROOT)), "sha256": sha256(s)}
                            for s in sources if str(Path(s).relative_to(ROOT)) not in known]
    manifest["oddL_added_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["oddL_generator"] = "paper_figures/fig_reentrant/prepare_data_oddL.py"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
