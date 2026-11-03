"""Add the odd-L (L = 31, 63) clean OBC cuts to the clean-frontier figure input.

Reads (read-only) the 18 complete campaigns clean-L{31,63}-oddL-fullgamma-h{11..19}-c128-v1
(full Gamma = 0-1.2, dGamma = 0.01, chi = 128) and stores per cut what the figure needs:
Gamma, bulk m_s^2 (window [L/4, 3L/4)), the L-1 finite H0 deaths of D = 1 - Cstag,
all H1 intervals of the Vietoris-Rips filtration (end-closure ring included), the
end-end and end-centre correlations Cstag(0, L-1) and Cstag(0, L//2), branch flag and bond,
and the local magnetizations <sigma^z> of the two ends and of the centre site, measured on the
selected MPS saved with each scan (selected_mps.h5).

Output: data_raw/clean_frontier_oddL_compact.npz. The existing source_manifest.json is
extended (not rewritten) with this file and its sources, so the even-L entries keep
their hashes. Run from the project root:
python paper_figures/fig_clean_frontier/prepare_data_oddL.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from prepare_data import h0_deaths, h1_diagram, sha256, OUT, ROOT

import sys
sys.path.insert(0, str(ROOT / "src"))
from clean_tfim_tn.io.clean_ising_mps_hdf5 import load_clean_ising_mps_collection  # noqa: E402

BASE = ROOT / "results" / "clean_ising_gamma_campaign"
SIZES = (31, 63)
H_VALUES = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9)
CHI = 128


# Additional odd-L cuts (2026-10-16): near the multicritical point (narrow window
# Gamma 0-0.2 or 0-0.4) and the low-field controls h = 0, 0.5 (L31).
EXTRA = {(63, 1.93): "clean-L63-oddL-cmp-h193-c128-v1",
         (63, 1.95): "clean-L63-oddL-cmp-pilot-h195-c128-v1",
         (63, 1.96): "clean-L63-oddL-cmp-h196-c128-v1",
         (63, 1.98): "clean-L63-oddL-cmp-h198-c128-v1",
         (31, 0.0): "clean-L31-oddL-fullgamma-h00-c128-v1",
         (31, 0.5): "clean-L31-oddL-fullgamma-h05-c128-v1"}


def scan_path(L, h):
    name = EXTRA.get((L, h), f"clean-L{L}-oddL-fullgamma-h{f'{h:.1f}'.replace('.', '')}-c128-v1")
    camp = BASE / name
    jobs = json.loads((camp / "campaign_status.json").read_text(encoding="utf-8"))["jobs"]
    if any(j["status"] != "complete" for j in jobs):
        raise RuntimeError(f"{camp.name} is not complete")
    return camp / "scans" / f"L{L}_h_{h:g}_J1_chi{CHI}" / "continuation_summary.npz"


def main():
    arrays, index, sources = {}, [], []
    for L, h in [(L, h) for L in SIZES for h in H_VALUES] + sorted(EXTRA):
        if True:
            path = scan_path(L, h)
            sources.append(path)
            with np.load(path, allow_pickle=False) as z:
                gamma = np.asarray(z["Gamma_values"], float)
                C = np.asarray(z["Cstag"], float)
                branch = np.asarray(z["branch_disagreement"], bool)
                bond = np.asarray(z["selected_max_bond_dimension"], int)
            a, b = L // 4, 3 * L // 4
            prefix = f"L{L}_h{h:g}_chi{CHI}"
            diagrams = [h1_diagram(m) for m in C]
            arrays[f"{prefix}_Gamma"] = gamma
            arrays[f"{prefix}_bulk_ms2"] = C[:, a:b, a:b].mean(axis=(1, 2))
            arrays[f"{prefix}_H0_deaths"] = np.stack([h0_deaths(m) for m in C])
            arrays[f"{prefix}_H1_counts"] = np.array([d.shape[0] for d in diagrams], int)
            arrays[f"{prefix}_H1_intervals"] = (np.concatenate(diagrams) if any(d.size for d in diagrams)
                                               else np.empty((0, 2)))
            arrays[f"{prefix}_C_end_end"] = C[:, 0, L - 1]
            arrays[f"{prefix}_C_end_centre"] = C[:, 0, L // 2]
            mps_path = path.parent / "selected_mps.h5"
            sources.append(mps_path)
            states = load_clean_ising_mps_collection(mps_path)["selected_states"]
            keys = sorted(states)
            if len(keys) != gamma.size:
                raise ValueError(f"{mps_path}: {len(keys)} states for {gamma.size} Gamma points")
            mz = np.array([np.real(states[k].expectation_value("Sigmaz")) for k in keys])
            arrays[f"{prefix}_mz_end_first"] = mz[:, 0]
            arrays[f"{prefix}_mz_end_last"] = mz[:, L - 1]
            arrays[f"{prefix}_mz_centre"] = mz[:, L // 2]
            arrays[f"{prefix}_branch_disagreement"] = branch
            arrays[f"{prefix}_max_bond"] = bond
            index.append({"L": L, "h": h, "chi": CHI, "prefix": prefix, "points": int(gamma.size),
                          "Gamma_range": [float(gamma.min()), float(gamma.max())],
                          "includes_full_gamma_scan": (L, h) not in EXTRA or h < 1.0, "extra": (L, h) in EXTRA, "sources": [str(path.relative_to(ROOT))]})
            print(prefix, gamma.size, flush=True)
    target = OUT / "clean_frontier_oddL_compact.npz"
    np.savez_compressed(target, index_json=np.array(json.dumps(index)), **arrays)
    manifest_path = OUT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [f for f in manifest["files"] if f["file"] != target.name]
    manifest["files"].append({"file": target.name, "sha256": sha256(target), "size_bytes": target.stat().st_size})
    known = {s["path"] for s in manifest["sources"]}
    manifest["sources"] += [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
                            for p in sources if str(p.relative_to(ROOT)) not in known]
    manifest["oddL_added_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["oddL_generator"] = "paper_figures/fig_clean_frontier/prepare_data_oddL.py"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("odd-L cuts:", len(index))


if __name__ == "__main__":
    main()
