"""B10 (review v3, R1-4): fidelity susceptibility in the reentrant regime (saved MPS, no DMRG).

R1: "the entry is abrupt because it is a quasi level crossing ... add chi_F in this regime to complete
the benchmark; it almost certainly peaks there as well."

For every reentrant realization of Fig. 6 (OBC, fixed_mean, chi = 32; L16 h=2.5 and L32 h=2.3125 from
reentrant-exit-anatomy-v1, L17 h=2.5 and L33 h=2.3125 from reentrant-oddL-L17-L33-v1) the selected
MPS of each Gamma0 point are loaded and

    chi_F(Gamma_mid) = -2 ln |<psi(Gamma_k)|psi(Gamma_{k+1})>| / (Gamma_{k+1} - Gamma_k)^2

is evaluated between consecutive grid points (the Gamma0 grid of the figure). Flags are not masked,
as for all estimators of Sec. VI. Entry and exit are compared with the physical zeros of C(r),
r = floor(L/2) - 1 (figure data), with the rules of Fig. 6:
  global     entry = maximum over the scan without the first interval (frozen H0 rule),
             exit = second wave (maximum after the first minimum that follows the global maximum,
             realizations with an interior second maximum only)
  split      entry / exit = maximum left / right of the apex of C(r) (frozen long-range rule)
The slope |dC(r)/dGamma0| with the same two rules and a constant equal to the median are the
references. Validation: the stored Czz of the first point of every scan is recomputed from its MPS.
Run (tn_tda):  python scripts/benchmark_fidelity_reentrant_review_v3.py
Output: results/review_v3/reentrant-fidelity-v1/{chiF_L*.npz, summary.json}
"""

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIGDIR = ROOT / "paper_figures/fig_reentrant"
sys.path.insert(0, str(FIGDIR))
import common as cm  # noqa: E402

BASE = ROOT / "results" / "random_ising_gamma_ensemble"
OUT = ROOT / "results" / "review_v3" / "reentrant-fidelity-v1"
EVEN = {16: "L16-reentrant-all1000-complete-v1", 32: "L32-reentrant-h23125-all1000-complete-v1"}
ODD = {17: "L17-reentrant-h2500", 33: "L33-reentrant-h23125"}
WORKERS = 6
RNG = np.random.default_rng(20261102)


def scan_dirs(L):
    if L in EVEN:
        rows = csv.DictReader((BASE / EVEN[L] / "analysis" / "source_manifest.csv").open(encoding="utf-8"))
        return {int(r["seed"]): (ROOT / r["source_npz"]).parent for r in rows}
    rows = []
    for st in sorted(BASE.glob(f"{ODD[L]}-oddL-*/campaign_status.json")):
        for j in json.loads(st.read_text(encoding="utf-8"))["jobs"]:
            rows.append((int(j["seed"]), Path(j["target_directory"])))
    return dict(rows)


def seed_fidelity(args):
    from clean_tfim_tn.io.mps_hdf5 import load_mps_collection_hdf5

    scan, g_fig = Path(args[0]), np.asarray(args[1])
    if not scan.is_absolute():
        scan = ROOT / scan
    man = json.loads((scan / "selected_mps_manifest.json").read_text(encoding="utf-8"))
    states = load_mps_collection_hdf5(scan / man["mps_file"])["selected_states"]
    rec = man["point_records"]
    g = np.array([float(r["Gamma0"]) for r in rec])
    if not np.allclose(g, g_fig):
        raise ValueError(f"Gamma0 grid mismatch in {scan}")
    with np.load(scan / "continuation_summary.npz") as z:
        czz0 = np.asarray(z["Czz"][0], float)
    psi = [states[r["state_key"]] for r in rec]
    err = float(np.max(np.abs(psi[0].correlation_function("Sigmaz", "Sigmaz") - czz0)))
    ov = np.array([abs(complex(psi[k].overlap(psi[k + 1]))) for k in range(len(psi) - 1)])
    norm = np.array([abs(complex(p.overlap(p))) for p in psi])
    ov = ov / np.sqrt(norm[:-1] * norm[1:])
    return -2.0 * np.log(np.clip(ov, 1e-300, None)) / np.diff(g) ** 2, ov.min(), err


def rules(d, v):
    """Entry/exit of a midpoint curve v with the global (H0) and split (long-range) rules of Fig. 6."""
    dd = {**d, "H0_velocity": v, "LR_velocity_w0": v}
    wave, status = cm.second_wave(dd)
    return {"global_entry": cm.topological_entry(dd), "global_exit": np.where(status == "interior", wave, np.nan),
            "split_entry": cm.split_velocity(dd, 0)[0], "split_exit": cm.split_velocity(dd, 0)[1]}


def mae(est, ref, step):
    ok = np.isfinite(est)
    ae = np.abs(est[ok] - ref[ok])
    boot = ae[RNG.integers(0, ae.size, (4000, ae.size))].mean(1)
    return {"N": int(ok.sum()), "of": int(ref.size), "MAE": float(ae.mean()),
            "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "median_est": float(np.nanmedian(est)), "within_one_step": float(np.mean(ae <= step + 1e-9))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for L in (16, 17, 32, 33):
        d = dict(np.load(FIGDIR / "data_raw" / f"reentrant_L{L}.npz"))
        g, mid = d["Gamma0"], d["Gamma0_mid"]
        step = float(g[1] - g[0])
        f = OUT / f"chiF_L{L}.npz"
        if not f.exists():
            dirs = scan_dirs(L)
            with ProcessPoolExecutor(max_workers=WORKERS) as pool:
                res = list(pool.map(seed_fidelity, [(str(dirs[int(s)]), g) for s in d["seeds"]]))
            np.savez_compressed(f, seeds=d["seeds"], Gamma0_mid=mid, chiF=np.stack([r[0] for r in res]),
                                min_overlap=np.array([r[1] for r in res]), czz_err=np.array([r[2] for r in res]))
        z = np.load(f)
        chiF = z["chiF"]
        cR = d["curves_r"][:, list(d["odd_r"]).index(int(d["R"]))]
        slope = np.abs(np.diff(cR, axis=1)) / step
        ests = {"chiF": rules(d, chiF), "slope": rules(d, slope)}
        res = {"h": float(d["h"]), "n": int(d["seeds"].size), "step": step,
               "max_czz_err": float(z["czz_err"].max()), "min_overlap": float(z["min_overlap"].min()),
               "median_entry": float(np.median(d["entry"])), "median_exit": float(np.median(d["exit"]))}
        for name, e in ests.items():
            for k, v in e.items():
                res[f"{name}_{k}"] = mae(v, d["entry"] if k.endswith("entry") else d["exit"], step)
        res["constant_entry"] = mae(np.full(d["entry"].size, np.median(d["entry"])), d["entry"], step)
        res["constant_exit"] = mae(np.full(d["exit"].size, np.median(d["exit"])), d["exit"], step)
        # contrast: chi_F at the physical entry and exit relative to its median over the scan
        ie = np.clip(np.searchsorted(mid, d["entry"]) - 1, 0, mid.size - 1)
        ie = np.where(np.abs(mid[np.minimum(ie + 1, mid.size - 1)] - d["entry"]) < np.abs(mid[ie] - d["entry"]),
                      np.minimum(ie + 1, mid.size - 1), ie)
        ix = np.argmin(np.abs(mid[None, :] - d["exit"][:, None]), axis=1)
        n = np.arange(chiF.shape[0])
        med = np.median(chiF, axis=1)
        res["contrast_entry_median"] = float(np.median(chiF[n, ie] / med))
        res["contrast_exit_median"] = float(np.median(chiF[n, ix] / med))
        res["chiF_max_is_entry_interval"] = float(np.mean(np.abs(ests["chiF"]["global_entry"] - d["entry"]) <= step / 2 + 1e-9))
        summary[f"L{L}"] = res
        print(f"L{L}: entry chiF {res['chiF_global_entry']['MAE']:.3f} (split {res['chiF_split_entry']['MAE']:.3f}, "
              f"slope {res['slope_split_entry']['MAE']:.3f}); exit chiF {res['chiF_global_exit']['MAE']:.3f} "
              f"(split {res['chiF_split_exit']['MAE']:.3f}, const {res['constant_exit']['MAE']:.3f}); "
              f"contrast {res['contrast_entry_median']:.0f}/{res['contrast_exit_median']:.1f}; czz err {res['max_czz_err']:.1e}",
              flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
