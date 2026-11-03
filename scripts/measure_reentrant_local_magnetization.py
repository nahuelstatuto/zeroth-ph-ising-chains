"""Measure local magnetizations on the saved reentrant MPS (no DMRG).

For every physically selected reentrant seed of the OBC cohorts (L16 h=2.5,
L32 h=2.3125), load the saved selected MPS of each Gamma0 point and measure
<sigma^z_i> and <sigma^x_i>. The full <sigma^z_i sigma^z_j> matrix is
recomputed at three points per seed (first, middle, last) and compared with
the stored Czz to validate the state/point mapping.

Run with the tn_tda environment (TeNPy + h5py):
    python scripts/measure_reentrant_local_magnetization.py
Output: results/random_ising_gamma_ensemble/reentrant-exit-anatomy-v1/analysis/
        local_magnetization_L{16,32}.npz
"""

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BASE = ROOT / "results" / "random_ising_gamma_ensemble"
OUT = BASE / "reentrant-exit-anatomy-v1" / "analysis"
COHORTS = {16: "L16-reentrant-all1000-complete-v1", 32: "L32-reentrant-h23125-all1000-complete-v1"}
WORKERS = 6


def measure_seed(scan_dir):
    from clean_tfim_tn.io.mps_hdf5 import load_mps_collection_hdf5

    scan_dir = Path(scan_dir)
    manifest = json.loads((scan_dir / "selected_mps_manifest.json").read_text(encoding="utf-8"))
    states = load_mps_collection_hdf5(scan_dir / manifest["mps_file"])["selected_states"]
    with np.load(scan_dir / "continuation_summary.npz") as z:
        czz_stored = np.asarray(z["Czz"], float)
        gam = np.asarray(z["Gamma0_values"], float)
    records = manifest["point_records"]
    if not np.allclose([float(r["Gamma0"]) for r in records], gam):
        raise ValueError(f"Gamma0 order mismatch in {scan_dir}")
    mz, mx, err = [], [], 0.0
    check = {0, len(records) // 2, len(records) - 1}
    for i, rec in enumerate(records):
        psi = states[rec["state_key"]]
        mz.append(psi.expectation_value("Sigmaz"))
        mx.append(psi.expectation_value("Sigmax"))
        if i in check:
            czz = psi.correlation_function("Sigmaz", "Sigmaz")
            err = max(err, float(np.max(np.abs(czz - czz_stored[i]))))
    return np.asarray(mz), np.asarray(mx), err


def main():
    for L, cohort in COHORTS.items():
        cache = np.load(OUT / f"cache_L{L}.npz")
        sel = np.flatnonzero(cache["selected"])
        rows = {int(r["seed"]): r for r in csv.DictReader(
            (BASE / cohort / "analysis" / "source_manifest.csv").open(encoding="utf-8"))}
        dirs = [str((ROOT / rows[int(cache["seeds"][k])]["source_npz"]).parent) for k in sel]
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            results = list(pool.map(measure_seed, dirs))
        mz = np.stack([r[0] for r in results])
        mx = np.stack([r[1] for r in results])
        err = np.array([r[2] for r in results])
        print(f"L{L}: {len(sel)} seeds, max |Czz - stored| over checked points = {err.max():.2e}", flush=True)
        np.savez_compressed(OUT / f"local_magnetization_L{L}.npz", seeds=cache["seeds"][sel], sel=sel,
                            Gamma0=cache["Gamma0"], mz=mz, mx=mx, max_czz_err=err)


if __name__ == "__main__":
    main()
