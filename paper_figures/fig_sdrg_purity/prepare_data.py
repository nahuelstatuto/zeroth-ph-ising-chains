"""Build the hashed input of the SDRG cluster figure (review v3): H0 hierarchy against the
clusters of the strong-disorder RG at h = 0, Gamma0 = 1, exact free-fermion correlations.

Sources (read-only, project results):
  results/review_v1/sdrg-vs-h0-v1/hierarchy_per_realization.npz     box ensemble (fixed_mean),
                            300 realizations for L = 32-512, the data of the former Fig. 3(c)
  results/review_v3/sdrg-disorder-strength-v1/fixed_logmean_d*_L*.npz   power-law disorder of
                            strength delta, shifted to fixed mean logarithms, 300 realizations per
                            (delta, L); delta = 3 is excluded (fails the QC 0 <= C <= 1 for L >= 64)
Stored in data_raw/:
  sdrg_hierarchy_box.npz    per-realization (best-node Jaccard, dendrogram purity) for the H0
                            dendrogram, the nearest-neighbour dendrogram and the unrelated partition
  sdrg_purity_delta.npz     per-realization dendrogram purity, H0 and nearest-neighbour, and seeds,
                            for every (delta, L)
  source_manifest.json      SHA-256 of these files and of every source
Run once:  python prepare_data.py   (from this folder)
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "results").is_dir() and (p / "scripts").is_dir())
OUT = HERE / "data_raw"
HIER = ROOT / "results" / "review_v1" / "sdrg-vs-h0-v1" / "hierarchy_per_realization.npz"
DS = ROOT / "results" / "review_v3" / "sdrg-disorder-strength-v1"
DELTAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SIZES = (32, 64, 128, 256)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    h = np.load(HIER)
    np.savez_compressed(OUT / "sdrg_hierarchy_box.npz", **{k: h[k] for k in h.files})
    sources = [HIER]
    arrays = {"deltas": np.array(DELTAS), "sizes": np.array(SIZES)}
    for d in DELTAS:
        for L in SIZES:
            src = DS / f"fixed_logmean_d{d:g}_L{L}.npz"
            z = np.load(src)
            for key in ("H0_purity", "NN_purity", "seeds"):
                arrays[f"d{d:g}_L{L}_{key}"] = np.asarray(z[key])
            sources.append(src)
    np.savez_compressed(OUT / "sdrg_purity_delta.npz", **arrays)
    files = [OUT / "sdrg_hierarchy_box.npz", OUT / "sdrg_purity_delta.npz"]
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "generator": "paper_figures/fig_sdrg_purity/prepare_data.py (review v3)",
                "conventions": "OBC; h=0; Gamma0=1; C=Cstag raw; topology on D=1-C; Pauli operators; exact free fermions; "
                               "box = fixed_mean J in [1,2]; delta family = fixed mean logarithms",
                "files": [{"file": f.name, "sha256": sha256(f), "size_bytes": f.stat().st_size} for f in files],
                "sources": [{"path": str(s.relative_to(ROOT)), "sha256": sha256(s)} for s in sources]}
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("ok", [f.name for f in files])


if __name__ == "__main__":
    main()
