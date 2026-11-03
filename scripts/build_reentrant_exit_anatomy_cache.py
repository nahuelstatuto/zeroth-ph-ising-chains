"""Compact cache of the OBC reentrant cohorts (L16 h=2.5, L32 h=2.3125; 1000 seeds each).

Read-only over the per-seed continuation_summary.npz listed in the source
manifests of the frozen all-1000 analyses. Stores, per seed: Cstag (float32),
half-chain entropy, couplings J_i, transverse-field pattern Gamma_i/Gamma0,
energies of both branches, branch-disagreement and chi-saturation flags, plus
the frozen physical selection (entry/exit crossings, sign class).

No DMRG. Output (exploratory, outside paper_figures):
results/random_ising_gamma_ensemble/reentrant-exit-anatomy-v1/analysis/cache_L{16,32}.npz
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "random_ising_gamma_ensemble"
OUT = BASE / "reentrant-exit-anatomy-v1" / "analysis"
COHORTS = {
    16: {"dir": "L16-reentrant-all1000-complete-v1", "h": 2.5, "chi": 32},
    32: {"dir": "L32-reentrant-h23125-all1000-complete-v1", "h": 2.3125, "chi": 64},
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for L, spec in COHORTS.items():
        analysis = BASE / spec["dir"] / "analysis"
        manifest = list(csv.DictReader((analysis / "source_manifest.csv").open(encoding="utf-8")))
        selection = {int(r["seed"]): r for r in csv.DictReader((analysis / "physical_selection_all_1000.csv").open(encoding="utf-8"))}
        frozen = np.load(analysis / "analysis_arrays.npz")
        n = len(manifest)
        first = np.load(ROOT / manifest[0]["source_npz"])
        G = first["Gamma0_values"].size
        arrays = {
            "seeds": np.zeros(n, int), "cohort": np.empty(n, "<U12"),
            "Gamma0": np.asarray(first["Gamma0_values"], float),
            "Cstag": np.zeros((n, G, L, L), np.float32),
            "half_chain_entropy": np.zeros((n, G)),
            "J": np.zeros((n, L - 1)), "gamma_pattern": np.zeros((n, L)),
            "energy_forward": np.zeros((n, G)), "energy_backward": np.zeros((n, G)),
            "branch_disagreement": np.zeros((n, G), bool), "chi_saturated": np.zeros((n, G), bool),
            "selected": np.zeros(n, bool), "entry": np.full(n, np.nan), "exit": np.full(n, np.nan),
            "sign_class": np.empty(n, "<U16"), "high_contrast": np.zeros(n, bool),
        }
        for k, row in enumerate(manifest):
            path = ROOT / row["source_npz"]
            if sha256(path) != row["sha256"]:
                raise ValueError(f"hash mismatch {path}")
            with np.load(path) as z:
                seed = int(row["seed"])
                arrays["seeds"][k] = seed
                arrays["cohort"][k] = row["cohort"]
                arrays["Cstag"][k] = z["Cstag"]
                arrays["half_chain_entropy"][k] = z["half_chain_entropy"]
                arrays["J"][k] = z["J"]
                gam = np.asarray(z["Gamma"], float)
                g0 = np.asarray(z["Gamma0_values"], float)
                nz = g0 > 0
                arrays["gamma_pattern"][k] = np.median(gam[nz] / g0[nz, None], axis=0)
                arrays["energy_forward"][k] = z["forward_energy"]
                arrays["energy_backward"][k] = z["backward_energy"]
                arrays["branch_disagreement"][k] = z["branch_disagreement"]
                arrays["chi_saturated"][k] = (np.maximum(z["forward_max_bond_dimension"], z["backward_max_bond_dimension"]) >= spec["chi"])
                if not np.isclose(float(z["h"]), spec["h"]):
                    raise ValueError("h mismatch")
            sel = selection[seed]
            arrays["selected"][k] = sel["selected"] == "True"
            arrays["entry"][k] = float(sel["entry_crossing"]) if sel["entry_crossing"] not in ("", "nan") else np.nan
            arrays["exit"][k] = float(sel["exit_crossing"]) if sel["exit_crossing"] not in ("", "nan") else np.nan
            arrays["sign_class"][k] = sel["sign_class"]
            arrays["high_contrast"][k] = sel["high_contrast"] == "True"
        # consistency with the frozen analysis
        if not np.array_equal(arrays["seeds"], frozen["seeds"]):
            raise ValueError("seed order differs from the frozen analysis")
        phys = np.stack([np.diagonal(c, offset=L // 2 - 1, axis1=1, axis2=2).mean(axis=1) for c in arrays["Cstag"]])
        err = float(np.max(np.abs(phys - frozen["physical_curves"])))
        if err > 1e-5:
            raise ValueError(f"physical curve mismatch {err}")
        if not np.array_equal(np.flatnonzero(arrays["selected"]), frozen["selected_indices"]):
            raise ValueError("selection differs from the frozen analysis")
        np.savez_compressed(OUT / f"cache_L{L}.npz", **arrays)
        meta = {"L": L, "h": spec["h"], "chi": spec["chi"], "seeds": n, "selected": int(arrays["selected"].sum()),
                "physical_curve_max_abs_diff_vs_frozen": err, "source_manifest": str((analysis / "source_manifest.csv").relative_to(ROOT))}
        (OUT / f"cache_L{L}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(meta, flush=True)


if __name__ == "__main__":
    main()
