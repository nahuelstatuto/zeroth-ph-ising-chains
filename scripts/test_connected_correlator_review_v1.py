"""Referee 2 (2.2): connected correlator at h > 0, odd chains L = 31 (Fig. 5 cuts h = 1.1-1.9).

Part B of test_correlator_choice_review_v1.py with a fourth input,
  conn   D = 1 - (-1)^|i-j| (<Z_i Z_j> - <Z_i><Z_j>),
where <Z_i> is measured on the saved selected MPS (no DMRG). Validation: the stored Czz is
recomputed on three points per cut. Same QC mask, peak rule and reference boundary as the paper.
Run:  python scripts/test_connected_correlator_review_v1.py [L]   (L = 31 or 63)
Output: results/review_v1/correlator-choice-v1/connected_L{L}.json
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_estimators_clean_frontier_review_v1 as bf  # noqa: E402
from test_correlator_choice_review_v1 import deaths  # noqa: E402

L = int(sys.argv[1]) if len(sys.argv) > 1 else 31
OUT = ROOT / "results" / "review_v1" / "correlator-choice-v1" / f"connected_L{L}.json"
KINDS = ("stag", "raw", "conn")       # raw = unstaggered <Z_i Z_j>


def measure(scan_dir):
    import h5py
    from tenpy.tools import hdf5_io
    scan_dir = Path(scan_dir)
    manifest = json.loads((scan_dir / "selected_mps_manifest.json").read_text(encoding="utf-8"))
    with h5py.File(scan_dir / manifest["mps_file"], "r") as f:      # clean collection: states under /payload
        states = hdf5_io.load_from_hdf5(f["payload"])["selected_states"]
    with np.load(scan_dir / "continuation_summary.npz") as z:
        g = np.asarray(z["Gamma_values"], float)
        czz = np.asarray(z["Czz"], float)
        flag = np.asarray(z["branch_disagreement"], bool)
        bond = np.asarray(z["selected_max_bond_dimension"], int)
    recs = manifest["point_records"]
    key = next(k for k in ("Gamma", "Gamma0", "gamma") if k in recs[0])
    gr = np.array([float(r[key]) for r in recs])
    if not np.allclose(gr, g):
        raise ValueError(f"Gamma order mismatch in {scan_dir}")
    mz, err = [], 0.0
    for i, r in enumerate(recs):
        psi = states[r["state_key"]]
        mz.append(np.asarray(psi.expectation_value("Sigmaz"), float))
        if i in (0, len(recs) // 2, len(recs) - 1):
            err = max(err, float(np.abs(psi.correlation_function("Sigmaz", "Sigmaz") - czz[i]).max()))
    return g, czz, np.array(mz), flag, bond, err


def main():
    z = np.load(bf.FIG / "data_raw" / "clean_frontier_oddL_compact.npz")
    cuts = sorted((round(e["h"], 4), e) for e in json.loads(str(z["index_json"]))
                  if e["L"] == L and round(e["h"], 4) in bf.GRID_H)
    dirs = []
    for h, e in cuts:
        if len(e["sources"]) != 1:
            raise ValueError(f"expected one source per L{L} cut")
        dirs.append(str((ROOT / e["sources"][0]).parent))
    with ProcessPoolExecutor(max_workers=5) as pool:
        res = list(pool.map(measure, dirs))
    sign = (-1.0) ** np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    out = {"cuts": [], "max_czz_check_err": max(r[5] for r in res)}
    for (h, e), (g, czz, mz, flag, bond, _) in zip(cuts, res):
        bad = flag | (bond >= e["chi"])
        allowed = ~(bad | np.r_[False, bad[:-1]] | np.r_[bad[1:], False])
        ref = float(np.interp(h, bf.REF_H, bf.REF_G))
        row = {"h": h, "reference": ref}
        for kind in KINDS:
            dd = []
            for k in range(len(g)):
                if kind == "raw":
                    dd.append(deaths(1.0 - czz[k]))
                    continue
                c = czz[k] if kind == "stag" else czz[k] - np.outer(mz[k], mz[k])
                dd.append(deaths(1.0 - sign * c))
            x = bf.masked_peak(g, bf.velocity(g, np.array(dd), L), allowed)["x"]
            row[kind] = {"peak": float(x), "err": abs(float(x) - ref)}
        out["cuts"].append(row)
        print(h, row, flush=True)
    for kind in KINDS:
        errs = np.array([r[kind]["err"] for r in out["cuts"]])
        out[kind] = {"MAE": float(errs.mean()), "max": float(errs.max())}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in (*KINDS, "max_czz_check_err")}))


if __name__ == "__main__":
    main()
