"""Data of Fig. S2 (V1): magnetization profiles across the reentrant entry (referee 2, Sec. 4.4).

Three representative L = 32 realizations (h = 2.3125, OBC, fixed_mean), one per class of entry
found by scripts/analyze_reentrant_domain_walls_review_v1.py:
  odd    Delta M ~ -2, one interior reversed block of odd length, two walls annihilated
  disp   Delta M ~ 0, one reversed block of even length, number of walls unchanged (displacement)
  even   Delta M ~ 0, one reversed block of even length, two walls annihilated
The class statistics of all entries (L = 16, 17, 32, 33) are copied from
results/review_v1/reentrant-domain-walls-v1/summary.json. Profiles <sigma^z_i> at the two grid
points bracketing each entry come from the magnetizations measured on the saved MPS.
Output: data_raw/reentrant_entry_walls.npz (+ entry in data_raw/source_manifest.json).
Run:  python prepare_entry_walls.py   (from this folder)
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "results").is_dir() and (p / "scripts").is_dir())
ANAT = ROOT / "results" / "random_ising_gamma_ensemble" / "reentrant-exit-anatomy-v1" / "analysis"
WALLS = ROOT / "results" / "review_v1" / "reentrant-domain-walls-v1" / "summary.json"
OUT = HERE / "data_raw"
EXAMPLES = {"odd": 28, "disp": 30, "even": 19}          # indices among the selected L32 realizations


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cache = np.load(ANAT / "cache_L32.npz")
    mag = np.load(ANAT / "local_magnetization_L32.npz")
    sel = mag["sel"]
    g = cache["Gamma0"]
    entry = cache["entry"][sel]
    walls = json.loads(WALLS.read_text(encoding="utf-8"))
    arrays = {}
    for name, n in EXAMPLES.items():
        ib = int(np.searchsorted(g, entry[n]) - 1)
        ev = walls["L32"]["events"][n]
        arrays[f"{name}_before"] = mag["mz"][n, ib]
        arrays[f"{name}_after"] = mag["mz"][n, ib + 1]
        arrays[f"{name}_gamma"] = np.array([g[ib], g[ib + 1]])
        arrays[f"{name}_seed"] = np.array(int(cache["seeds"][sel][n]))
        arrays[f"{name}_dM"] = np.array(ev["dM"])
        arrays[f"{name}_walls"] = np.array([ev["walls_before"], ev["walls_after"]])
    stats = {}
    for L in (16, 17, 32, 33):
        for cls in ("dM~0", "dM~-2"):
            d = walls[f"L{L}"].get(cls)
            if d:
                stats[f"L{L}_{cls}"] = {"count": d["count"], "dW_hist": d["dW_hist"]}
    target = OUT / "reentrant_entry_walls.npz"
    np.savez_compressed(target, **arrays, stats_json=np.array(json.dumps(stats)))
    manifest_path = OUT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [f for f in manifest["files"] if f["file"] != target.name]
    manifest["files"].append({"file": target.name, "sha256": sha256(target), "size_bytes": target.stat().st_size})
    known = {s["path"] for s in manifest["sources"]}
    for s in (ANAT / "local_magnetization_L32.npz", WALLS):
        if str(s.relative_to(ROOT)) not in known:
            manifest["sources"].append({"path": str(s.relative_to(ROOT)), "sha256": sha256(s)})
    manifest["entry_walls_added_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print({k: arrays[f"{k}_walls"].tolist() for k in EXAMPLES})


if __name__ == "__main__":
    main()
