"""Disorder-averaged C(r) of rings and open chains for Fig. 7(c) (review v3, R1-4: wall parity of the lobe).

All realizations (no selection), h = 2.5, r = floor(L/2) - 1 = 7, Gamma0 = 0-0.6 in 49 points.
  rings (PBC, ED):  main cohort of pbc-reentrant-ED-parity-v1, seeds 5000-5999, L = 16 and 17,
                    C = Cstag with circular phase, loaded by scripts/analyze_pbc_reentrant_ED_parity.load
                    (refuses duplicates and missing seeds); C(r) by run_pbc_reentrant_ED_parity.c_of_r.
  open (OBC, DMRG): the production scans of Fig. 7, chi = 32, L16 (1000 seeds,
                    L16-reentrant-all1000-complete-v1) and L17 (400 seeds, L17-reentrant-h2500-oddL-*),
                    selected branch of continuation_summary.npz, C(r) = (-1)^r mean_i Czz[i, i+r].
Per-seed curves are stored, so the figure computes the means and bootstrap bands itself.
Output: data_raw/reentrant_parity_average.npz; source_manifest.json is extended, not rewritten.
Run from the project root:  python paper_figures/fig_reentrant/prepare_parity_average.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "results").is_dir())
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from analyze_pbc_reentrant_ED_parity import load as load_pbc  # noqa: E402
from run_pbc_reentrant_ED_parity import GRID as PBC_GRID, c_of_r  # noqa: E402
from benchmark_fidelity_reentrant_review_v3 import scan_dirs  # noqa: E402

OUT = HERE / "data_raw"
H = 2.5
PBC_SEEDS = range(5000, 6000)
SIZES = (16, 17)


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def obc_curves(L):
    r = L // 2 - 1
    dirs = scan_dirs(L)
    seeds = np.array(sorted(dirs))
    curves, grid = [], None
    for s in seeds:
        with np.load(Path(dirs[s]) / "continuation_summary.npz") as z:
            g = np.asarray(z["Gamma0_values"], float)
            if not np.isclose(float(z["h"]), H):
                raise ValueError(f"L{L} seed {s}: h = {float(z['h'])}")
            czz = np.asarray(z["Czz"], float)
        if grid is None:
            grid = g
        elif not np.allclose(g, grid):
            raise ValueError(f"L{L} seed {s}: different Gamma0 grid")
        curves.append((-1) ** r * np.mean([czz[:, i, i + r] for i in range(L - r)], axis=0))
    return seeds, grid, np.array(curves), sorted({str(Path(dirs[s]).parent.relative_to(ROOT)) for s in seeds})


def main():
    out = {}
    sources = []
    for L in SIZES:
        d = load_pbc(L, H, set(PBC_SEEDS), partial=False)
        out[f"pbc_L{L}_seeds"] = d["seeds"]
        out[f"pbc_L{L}_C"] = c_of_r(d["C"], L // 2 - 1)
        seeds, g, curves, dirs = obc_curves(L)
        if not np.allclose(g, PBC_GRID):
            raise ValueError("open and periodic grids differ")
        out[f"obc_L{L}_seeds"], out[f"obc_L{L}_C"] = seeds, curves
        sources += dirs
        print(L, "rings", d["seeds"].size, "open", seeds.size, flush=True)
    target = OUT / "reentrant_parity_average.npz"
    np.savez_compressed(target, Gamma0=np.asarray(PBC_GRID, float), h=H, R=7, **out)
    manifest_path = OUT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [f for f in manifest["files"] if f["file"] != target.name]
    manifest["files"].append({"file": target.name, "sha256": sha256(target), "size_bytes": target.stat().st_size})
    manifest["parity_average_sources"] = {
        "rings": "results/random_ising_gamma_ensemble/pbc-reentrant-ED-parity-L{16,17}-h25000-s*-v1 (seeds 5000-5999)",
        "open": sorted(set(sources))}
    manifest["parity_average_added_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["parity_average_generator"] = "fig_reentrant/prepare_parity_average.py"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
