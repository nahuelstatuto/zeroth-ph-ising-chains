"""Referee 1, round 2 (3.2): xi_lambda with the connected staggered correlator for even chains (L = 64).

Same cuts, merge rule, QC mask, peak rule and reference boundary as the benchmark of Table I
(benchmark_estimators_clean_frontier_review_v1.load): cuts with h < 2J(1 - 1/L), windows of each cut
merged by keeping the lower-energy state at duplicated fields. <sigma^z_i> is measured on the stored
selected MPS of every window (no DMRG), and
  connected staggered  C^c_ij = (-1)^|i-j| (<Z_i Z_j> - <Z_i><Z_j>) = Cstag_ij - (-1)^|i-j| <Z_i><Z_j>.
Validation: the raw maxima must reproduce the xi MAE of Table I for L = 64; the stored Czz is
recomputed on three states per window.
Run (tn_tda):  python scripts/test_connected_correlator_even_review_v2.py
Output: results/review_v2/connected-even-v1/summary.json
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_estimators_clean_frontier_review_v1 as bf  # noqa: E402

OUT = ROOT / "results" / "review_v2" / "connected-even-v1"
L = 64


def measure(src):
    import h5py
    from tenpy.tools import hdf5_io
    path = Path(src) if Path(src).is_absolute() else ROOT / src
    d = path.parent
    manifest = json.loads((d / "selected_mps_manifest.json").read_text(encoding="utf-8"))
    with h5py.File(d / manifest["mps_file"], "r") as f:
        states = hdf5_io.load_from_hdf5(f["payload"])["selected_states"]
    with np.load(path) as a:
        g = np.asarray(a["Gamma_values"], float)
        czz = np.asarray(a["Czz"], float)
        scan = {"gamma": g, "cstag": np.asarray(a["Cstag"], float), "energy": np.asarray(a["selected_energy_density"], float),
                "bond": np.asarray(a["selected_max_bond_dimension"], int), "branch": np.asarray(a["branch_disagreement"], bool)}
    recs = manifest["point_records"]
    if not np.allclose([float(r["Gamma"]) for r in recs], g):
        raise ValueError(f"Gamma order mismatch in {d}")
    mz, err = [], 0.0
    for i, r in enumerate(recs):
        psi = states[r["state_key"]]
        mz.append(np.asarray(psi.expectation_value("Sigmaz"), float))
        if i in (0, len(recs) // 2, len(recs) - 1):
            err = max(err, float(np.abs(psi.correlation_function("Sigmaz", "Sigmaz") - czz[i]).max()))
    scan["mz"] = np.array(mz)
    return str(src), scan, err


def merge(scans):
    pts = {}
    for s in scans:
        for i, gam in enumerate(s["gamma"]):
            key = round(float(gam), 12)
            p = {k: s[k][i] for k in ("cstag", "energy", "bond", "branch", "mz")}
            if key not in pts or p["energy"] < pts[key]["energy"] and not np.isclose(p["energy"], pts[key]["energy"], rtol=1e-9, atol=1e-11):
                pts[key] = p
    keys = sorted(pts)
    return np.array(keys), {k: np.array([pts[q][k] for q in keys]) for k in pts[keys[0]]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(bf.FIG / "data_raw" / "clean_frontier_compact.npz")
    best = {}
    for e in json.loads(str(z["index_json"])):
        key = (e["L"], round(e["h"], 4))
        if e["L"] == L and (key not in best or e["chi"] > best[key]["chi"]):
            best[key] = e
    cuts = {k: e for k, e in sorted(best.items()) if k not in bf.QC_CUTS and k[1] < 2 * (1 - 1 / L) - 1e-12}
    sources = sorted({s for e in cuts.values() for s in e["sources"]})
    with ProcessPoolExecutor(max_workers=6) as pool:
        measured = {src: (scan, err) for src, scan, err in pool.map(measure, sources)}
    sign = (-1.0) ** np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    rows, errs = [], {"raw": [], "conn": []}
    for (LL, h), e in cuts.items():
        g, d = merge([measured[s][0] for s in e["sources"]])
        if g.size < 4:
            continue
        flag = d["branch"] | (d["bond"] >= e["chi"])
        allowed = ~(flag | np.r_[False, flag[:-1]] | np.r_[flag[1:], False])
        ref = float(np.interp(h, bf.REF_H, bf.REF_G))
        row = {"h": h, "reference": ref}
        for kind in ("raw", "conn"):
            C = d["cstag"] if kind == "raw" else d["cstag"] - sign[None] * np.einsum("ki,kj->kij", d["mz"], d["mz"])
            deaths = np.stack([bf.frontier_deaths(m) for m in C])
            x = bf.masked_peak(g, bf.velocity(g, deaths, L), allowed)["x"]
            row[kind] = {"peak": x, "err": abs(x - ref)}
            errs[kind].append(abs(x - ref))
        row["max_abs_mz_staggered_centre"] = float(np.abs(d["mz"][:, L // 4:3 * L // 4] * sign[0, L // 4:3 * L // 4]).mean(axis=1).max())
        rows.append(row)
        print(h, json.dumps({k: row[k] for k in ("raw", "conn")}), flush=True)
    out = {"L": L, "cuts": rows, "max_czz_check_err": max(v[1] for v in measured.values()),
           **{k: {"MAE": float(np.mean(v)), "max": float(np.max(v)), "n": len(v)} for k, v in errs.items()}}
    ref = json.loads((ROOT / "results" / "review_v1" / "benchmark-clean-frontier-v1" / "summary.json").read_text())["per_size"]["L64"]["xi"]["MAE"]
    out["benchmark_raw_MAE"] = ref
    if abs(out["raw"]["MAE"] - ref) > 1e-6:
        print(f"WARNING raw MAE {out['raw']['MAE']} differs from Table I {ref}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("raw", "conn", "benchmark_raw_MAE", "max_czz_check_err")}))


if __name__ == "__main__":
    main()
