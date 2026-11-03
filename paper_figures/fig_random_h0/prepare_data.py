"""Build the hashed input of Fig. 3 (V1): random chain at h = 0, exact free fermions.

Sources (read-only, project results):
  results/review_v1/free-fermion-h0-v1/random_L{16,...,256}.npz       exact campaign (1000 realizations,
                                                                      500 for L = 256), grid step 0.005
  results/review_v1/free-fermion-h0-v1/fidelity_random_L*.npz         not used here
  results/review_v1/sdrg-vs-h0-v1/hierarchy_per_realization.npz       H0 hierarchy vs SDRG clusters
Stored in data_raw/:
  random_h0_peaks.npz       per-realization maxima of xi_lambda (closed form, Eq. 6) and of the
                            physical response -d m2_bulk / dGamma0 (interior maximum, quadratic
                            refinement; maxima at the grid ends set to NaN)
  random_h0_sdrg_hierarchy.npz   per-realization (best-node Jaccard, dendrogram purity) at Gamma0 = 1
                            for the H0 dendrogram, the nearest-neighbour dendrogram and the
                            unrelated-partition baseline, L = 32-512
  source_manifest.json      SHA-256 of these files and of every source
Run once:  python prepare_data.py   (from this folder)
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "results").is_dir() and (p / "scripts").is_dir())
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_free_fermion_h0_review_v1 as an  # noqa: E402

OUT = HERE / "data_raw"
SIZES = (16, 32, 64, 128, 256)
HIER = ROOT / "results" / "review_v1" / "sdrg-vs-h0-v1" / "hierarchy_per_realization.npz"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    arrays, sources = {"sizes": np.array(SIZES)}, []
    for L in SIZES:
        src = an.DATA / f"random_L{L}.npz"
        z = np.load(src)
        g, mid, cv = an.curves(z)
        for key in ("xi", "phys"):
            p, edge = an.peaks(g, cv[key])
            arrays[f"L{L}_{key}"] = np.where(edge, np.nan, p)
        arrays[f"L{L}_seeds"] = np.asarray(z["seeds"])
        sources.append(src)
    np.savez_compressed(OUT / "random_h0_peaks.npz", **arrays)
    h = np.load(HIER)
    np.savez_compressed(OUT / "random_h0_sdrg_hierarchy.npz", **{k: h[k] for k in h.files})
    sources.append(HIER)
    files = [OUT / "random_h0_peaks.npz", OUT / "random_h0_sdrg_hierarchy.npz"]
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "generator": "paper_figures/fig_random_h0/prepare_data.py (V1)",
                "conventions": "OBC; fixed_mean; C=Cstag raw; topology on D=1-C; Pauli operators; h=0 exact free fermions",
                "files": [{"file": f.name, "sha256": sha256(f), "size_bytes": f.stat().st_size} for f in files],
                "sources": [{"path": str(s.relative_to(ROOT)), "sha256": sha256(s)} for s in sources]}
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("ok", [f.name for f in files])


if __name__ == "__main__":
    main()
