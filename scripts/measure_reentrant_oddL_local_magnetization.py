"""Measure local magnetizations on the saved odd-L reentrant MPS (no DMRG).

Odd-L counterpart of measure_reentrant_local_magnetization.py. For every physically
selected reentrant seed of the OBC odd cohorts (L17 h=2.5, L33 h=2.3125; selection of
analyze_reentrant_oddL_L17_L33.py), load the saved selected MPS of each Gamma0 point and
measure <sigma^z_i> and <sigma^x_i>. Validation: <sigma^z_i> is compared with the mz
already stored in cache_L{L}.npz, and the full <sigma^z_i sigma^z_j> matrix is recomputed
at three points per seed and compared with the stored Czz.
The transverse-field pattern Gamma_i / Gamma0 (fixed along the scan) is stored for
-dE/dGamma0 = sum_i (Gamma_i / Gamma0) <sigma^x_i> (Hellmann-Feynman).

Run with the tn_tda environment (TeNPy + h5py):
    python scripts/measure_reentrant_oddL_local_magnetization.py
Output: results/random_ising_gamma_ensemble/reentrant-oddL-L17-L33-v1/analysis/
        local_magnetization_L{17,33}.npz
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BASE = ROOT / "results" / "random_ising_gamma_ensemble"
OUT = BASE / "reentrant-oddL-L17-L33-v1" / "analysis"
COHORTS = {17: "L17-reentrant-h2500", 33: "L33-reentrant-h23125"}
WORKERS = 6


def scan_dirs(prefix):
    rows = []
    for st in sorted(BASE.glob(f"{prefix}-oddL-*/campaign_status.json")):
        for j in json.loads(st.read_text(encoding="utf-8"))["jobs"]:
            rows.append((int(j["seed"]), Path(j["target_directory"])))
    return dict(sorted(rows))


def measure_seed(scan_dir):
    from clean_tfim_tn.io.mps_hdf5 import load_mps_collection_hdf5

    scan_dir = Path(scan_dir)
    manifest = json.loads((scan_dir / "selected_mps_manifest.json").read_text(encoding="utf-8"))
    states = load_mps_collection_hdf5(scan_dir / manifest["mps_file"])["selected_states"]
    with np.load(scan_dir / "continuation_summary.npz") as z:
        czz_stored = np.asarray(z["Czz"], float)
        gam = np.asarray(z["Gamma0_values"], float)
        Gamma = np.asarray(z["Gamma"], float)
    records = manifest["point_records"]
    if not np.allclose([float(r["Gamma0"]) for r in records], gam):
        raise ValueError(f"Gamma0 order mismatch in {scan_dir}")
    nz = gam > 0
    pattern = np.median(Gamma[nz] / gam[nz, None], axis=0)
    if not np.allclose(Gamma[nz] / gam[nz, None], pattern[None, :], atol=1e-10):
        raise ValueError(f"transverse-field pattern not fixed along the scan in {scan_dir}")
    mz, mx, err = [], [], 0.0
    check = {0, len(records) // 2, len(records) - 1}
    for i, rec in enumerate(records):
        psi = states[rec["state_key"]]
        mz.append(psi.expectation_value("Sigmaz"))
        mx.append(psi.expectation_value("Sigmax"))
        if i in check:
            czz = psi.correlation_function("Sigmaz", "Sigmaz")
            err = max(err, float(np.max(np.abs(czz - czz_stored[i]))))
    return np.asarray(mz), np.asarray(mx), pattern, err


def main():
    for L, prefix in COHORTS.items():
        cache = np.load(OUT / f"cache_L{L}.npz")
        sel = cache["selected"]
        seeds = cache["seeds"][sel]
        dirs = scan_dirs(prefix)
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            results = list(pool.map(measure_seed, [str(dirs[int(s)]) for s in seeds]))
        mz = np.stack([r[0] for r in results])
        mx = np.stack([r[1] for r in results])
        pattern = np.stack([r[2] for r in results])
        err = np.array([r[3] for r in results])
        mz_err = float(np.abs(mz - cache["mz"]).max())
        print(f"L{L}: {len(seeds)} seeds, max |Czz - stored| = {err.max():.2e}, "
              f"max |mz - cached mz| = {mz_err:.2e}", flush=True)
        if err.max() > 1e-8 or mz_err > 1e-8:
            raise ValueError("validation against stored data failed")
        np.savez_compressed(OUT / f"local_magnetization_L{L}.npz", seeds=seeds, sel=sel,
                            Gamma0=cache["Gamma0"], mz=mz, mx=mx, gamma_pattern=pattern,
                            max_czz_err=err, max_mz_err=mz_err)


if __name__ == "__main__":
    main()
