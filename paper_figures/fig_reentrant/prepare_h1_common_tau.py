"""H1 interval counts of the reentrant cohorts at the threshold shared with the clean frontier.

The clean frontier (Sec. V) uses the first long-lived H1 cycle with lifetime above
tau = 0.04. For the reentrant figure the same threshold and rule are applied after the
physical entry. Counts are built from the H1 lifetimes computed by
scripts/explore_h1_tau_common.py (results/random_ising_gamma_ensemble/h1-tau-common-v1/),
which also documents that no threshold between 0.004 and 0.08 locates the exit.
Output: data_raw/reentrant_H1_common_tau.npz; source_manifest.json is extended.
Run from the project root:  python paper_figures/fig_reentrant/prepare_h1_common_tau.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_data import ROOT, OUT, BASE, sha256  # noqa: E402

TAU = 0.04
SRC = BASE / "h1-tau-common-v1"


def main():
    arrays, sources = {"tau": np.array(TAU)}, []
    for L in (16, 17, 32, 33):
        path = SRC / f"lifetimes_L{L}.npz"
        z = np.load(path, allow_pickle=True)
        fig = np.load(OUT / f"reentrant_L{L}.npz")
        if not np.allclose(z["entry"], fig["entry"]) or not np.allclose(z["exit"], fig["exit"]):
            raise ValueError(f"realization order mismatch for L{L}")
        arrays[f"L{L}_H1_count"] = np.array([[int(np.sum(np.asarray(x) > TAU)) for x in per] for per in z["life"]])
        sources.append(path)
    target = OUT / "reentrant_H1_common_tau.npz"
    np.savez_compressed(target, **arrays)
    manifest_path = OUT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [f for f in manifest["files"] if f["file"] != target.name]
    manifest["files"].append({"file": target.name, "sha256": sha256(target), "size_bytes": target.stat().st_size})
    known = {s["path"] for s in manifest["sources"]}
    manifest["sources"] += [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in sources
                            if str(p.relative_to(ROOT)) not in known]
    manifest["h1_common_tau_added_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("ok", {k: v.shape for k, v in arrays.items() if k != "tau"})


if __name__ == "__main__":
    main()
