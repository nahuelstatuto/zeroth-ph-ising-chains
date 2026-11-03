"""Data of Fig. S3 (V1): the first H1 cycle drawn in one cut (referee 2).

Clean odd chain L = 31, h = 1.7 J, OBC, chi = 128, the full-range scan of Fig. 4 (same source as
data_raw/clean_frontier_oddL_compact.npz). Copies the staggered correlation matrices C = Cstag of
the whole cut, with the quality flags, so that the figure can recompute the Rips persistence and the
representative cycle from raw data.
Output: data_raw/clean_frontier_h1_cycle_L31.npz (+ entry in data_raw/source_manifest.json).
Run:  python prepare_h1_cycle.py   (from this folder)
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "results").is_dir() and (p / "scripts").is_dir())
OUT = HERE / "data_raw"
L, H = 31, 1.7


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    z = np.load(OUT / "clean_frontier_oddL_compact.npz")
    e = [e for e in json.loads(str(z["index_json"])) if e["L"] == L and abs(e["h"] - H) < 1e-9]
    if len(e) != 1 or len(e[0]["sources"]) != 1:
        raise SystemExit("expected a single source scan for the cut")
    e = e[0]
    src = ROOT / e["sources"][0]
    with np.load(src) as a:
        arrays = {"Gamma": np.asarray(a["Gamma_values"], float), "Cstag": np.asarray(a["Cstag"], float),
                  "branch_disagreement": np.asarray(a["branch_disagreement"], bool),
                  "max_bond": np.asarray(a["selected_max_bond_dimension"], int)}
    arrays.update(L=np.array(L), h=np.array(H), chi=np.array(e["chi"]))
    target = OUT / "clean_frontier_h1_cycle_L31.npz"
    np.savez_compressed(target, **arrays)
    manifest_path = OUT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [f for f in manifest["files"] if f["file"] != target.name]
    manifest["files"].append({"file": target.name, "sha256": sha256(target), "size_bytes": target.stat().st_size})
    known = {s["path"] for s in manifest["sources"]}
    if str(src.relative_to(ROOT)) not in known:
        manifest["sources"].append({"path": str(src.relative_to(ROOT)), "sha256": sha256(src)})
    manifest["h1_cycle_added_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(target.name, arrays["Cstag"].shape)


if __name__ == "__main__":
    main()
