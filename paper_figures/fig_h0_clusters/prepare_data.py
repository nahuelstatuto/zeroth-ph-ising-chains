"""Build the hashed input of the schematic figure on H0 clusters (V1).

Didactic open chain of L = 12 sites (1-based numbering in the figure) with chosen couplings,
h = 0: two halves {4,5,6} and {8,9,10} with J = 3 and Gamma = 0.6; site 7 with Gamma_7 = 5 and
J = 1.5 to both halves; other sites J = 0.5, Gamma = 1.5. The exact ground state is computed
with the free-fermion solution (clean_tfim_tn.exact.free_fermion_h0) and the clusters with the
strong-disorder RG of scripts/compare_sdrg_h0_clusters_review_v1.py.
Stored in data_raw/toy_chain.npz: J, Gamma, C = Cstag, SDRG labels; source_manifest.json.
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from clean_tfim_tn.exact.free_fermion_h0 import cstag_elimination, majorana_state  # noqa: E402
from compare_sdrg_h0_clusters_review_v1 import sdrg_partition  # noqa: E402

OUT = HERE / "data_raw"
L = 12


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    J = np.full(L - 1, 0.5)
    G = np.full(L, 1.5)
    for b in (3, 4, 7, 8):           # 0-based bonds 3-4, 4-5, 7-8, 8-9
        J[b] = 3.0
    J[5] = J[6] = 1.5                # bonds to the strong-field site (0-based 6 = site 7)
    for s in (3, 4, 5, 7, 8, 9):
        G[s] = 0.6
    G[6] = 5.0
    C = cstag_elimination(majorana_state(J, G)[0])
    labels = sdrg_partition(J, G)
    target = OUT / "toy_chain.npz"
    np.savez_compressed(target, J=J, Gamma=G, C=C, sdrg_labels=labels)
    sources = [ROOT / "src" / "clean_tfim_tn" / "exact" / "free_fermion_h0.py",
               ROOT / "scripts" / "compare_sdrg_h0_clusters_review_v1.py"]
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "generator": "paper_figures/fig_h0_clusters/prepare_data.py (V1)",
                "conventions": "OBC; h=0; C=Cstag raw; topology on D=1-C; exact free fermions; didactic couplings",
                "files": [{"file": target.name, "sha256": sha256(target), "size_bytes": target.stat().st_size}],
                "sources": [{"path": str(s.relative_to(ROOT)), "sha256": sha256(s)} for s in sources]}
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("SDRG labels", labels.tolist(), "C(6,8) =", round(C[5, 7], 3), "C(6,7) =", round(C[5, 6], 3))


if __name__ == "__main__":
    main()
