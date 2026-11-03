"""Build the hashed input of Fig. 2 (V1): clean chain at h = 0, exact free fermions.

Sources (read-only, project results):
  results/review_v1/free-fermion-h0-v1/clean_L{16,32,64,128,256}.npz   (grid g = 0.5-1.5, step 0.005)
  results/review_v1/free-fermion-h0-v1/clean_L512_full.npz             (same grid)
Stored in data_raw/clean_h0_exact.npz, per size: g, sorted finite H0 deaths of D = 1 - Cstag
(minimum spanning tree) and the bulk squared staggered magnetization m2_bulk (central half);
SHA-256 of the file and of the sources in data_raw/source_manifest.json.
Review v3 (B5): data_raw/clean_h0_refined_peaks.npz holds the maxima of xi_lambda and of the
physical response refined on a local grid of step 1e-4 (xi) and by bounded maximization (physical),
from results/review_v3/clean-fss-systematics-v1/summary.json
(scripts/analyze_clean_fss_systematics_review_v3.py). Panel (c) and the fits use these maxima,
because the 0.005 grid biases the maxima of the larger chains by up to 6e-4. The DMRG data of the
v0 figure (L = 8-128) are kept in old_v0/ and used as validation in Appendix A.
Run once:  python prepare_data.py   (from this folder)
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "results").is_dir() and (p / "scripts").is_dir())
SRC = ROOT / "results" / "review_v1" / "free-fermion-h0-v1"
REFINED = ROOT / "results" / "review_v3" / "clean-fss-systematics-v1" / "summary.json"
OUT = HERE / "data_raw"
SIZES = (16, 32, 64, 128, 256, 512)


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
        src = SRC / ("clean_L512_full.npz" if L == 512 else f"clean_L{L}.npz")
        z = np.load(src)
        deaths = np.asarray(z["deaths"], float)
        deaths = deaths[0] if deaths.ndim == 3 else deaths
        m2 = np.asarray(z["m2_bulk"], float)
        m2 = m2[0] if m2.ndim == 2 else m2
        arrays[f"L{L}_g"] = np.asarray(z["Gamma0"], float)
        arrays[f"L{L}_deaths"] = deaths
        arrays[f"L{L}_m2_bulk"] = m2
        sources.append(src)
    target = OUT / "clean_h0_exact.npz"
    np.savez_compressed(target, **arrays)
    peaks = json.loads(REFINED.read_text(encoding="utf-8"))["peaks"]
    refined = OUT / "clean_h0_refined_peaks.npz"
    np.savez_compressed(refined, sizes=np.array(SIZES),
                        xi=np.array([peaks["xi"][str(L)] for L in SIZES]),
                        physical=np.array([peaks["phys"][str(L)] for L in SIZES]))
    sources.append(REFINED)
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "generator": "paper_figures/fig_clean_h0/prepare_data.py (V1, refined maxima added in review v3)",
                "conventions": "OBC; h=0; clean J=1; C=Cstag raw; topology on D=1-C; exact free fermions",
                "files": [{"file": f.name, "sha256": sha256(f), "size_bytes": f.stat().st_size} for f in (target, refined)],
                "sources": [{"path": str(s.relative_to(ROOT)), "sha256": sha256(s)} for s in sources]}
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("ok", {L: arrays[f"L{L}_g"].size for L in SIZES})


if __name__ == "__main__":
    main()
