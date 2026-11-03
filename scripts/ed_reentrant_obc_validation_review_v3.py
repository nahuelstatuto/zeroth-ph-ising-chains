"""C2 (review v3, R1-4): exact diagonalization of the open reentrant chains of Fig. 6 (L = 16, 17).

R1: "with chi = 32 at L = 16 and 17 and abrupt transitions between wall configurations, DMRG can stay
in metastable states. ED with Lanczos is feasible for L = 16 and 17, at least for a subset of
realizations. Extend it to the open chains of Fig. 6."

For every reentrant realization of Fig. 6 with L = 16 (h = 2.5, 97) and L = 17 (h = 2.5, 40) the exact
couplings J_b and fields Gamma_i(Gamma0) of each point are read from its DMRG scan
(continuation_summary.npz), and

    H = sum_b J_b Z_b Z_{b+1} - sum_i Gamma_i X_i - h sum_i Z_i        (OBC, Pauli operators)

is diagonalized at every Gamma0 of the scan with Lanczos (two lowest states, independent random start
at every point, so that no branch is followed across an avoided crossing). Gamma0 = 0 is diagonal and
uses the uniform mixture of the degenerate classical minima. Compared with the DMRG state of the
figure (selected branch):
  dE        E_DMRG - E_0 (selected branch energy; >= 0 up to the Lanczos tolerance)
  gap       E_1 - E_0
  dC        max_ij |Czz_ED - Czz_DMRG| and |C_ED(r) - C_DMRG(r)|, r = floor(L/2) - 1
  entry/exit zeros of C_ED(r) (rule of Fig. 6, fig_reentrant/common.entry_exit) against the DMRG ones
  reentrant whether C_ED(r) passes the frozen selection rule (select_physical_seeds)
The operators and their sign convention are checked against clean_tfim_tn.ed.random_ising
(build_random_ising_hamiltonian) and against the DMRG energies.
Run (tn_tda):  python scripts/ed_reentrant_obc_validation_review_v3.py
Output: results/review_v3/ed-reentrant-obc-v1/{ed_L16.npz, ed_L17.npz, summary.json}
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_fidelity_reentrant_review_v3 import FIGDIR, cm, scan_dirs  # noqa: E402

OUT = ROOT / "results" / "review_v3" / "ed-reentrant-obc-v1"
SIZES = (16, 17)
WORKERS = 12
TOL = 1e-12
SELECTION = {"endpoint_points": 2, "negative_margin": 0.02, "positive_amplitude": 0.05,
             "minimum_lobe_width": 0.10, "minimum_positive_run": 2}  # frozen rule (Appendix A)


def operators(L, J, h):
    states = np.arange(2 ** L)
    z = (1 - 2 * ((states[:, None] >> (L - 1 - np.arange(L))[None, :]) & 1)).astype(np.int8)
    diag = sum(J[b] * z[:, b].astype(float) * z[:, b + 1] for b in range(L - 1)) - h * z.sum(axis=1)
    flips = [sparse.csr_matrix((np.ones(states.size), (states, states ^ (1 << (L - 1 - i)))),
                               shape=(states.size, states.size)) for i in range(L)]
    return z, diag, flips


def realization(args):
    L, scan = args
    with np.load(Path(scan) / "continuation_summary.npz") as zf:
        g = np.asarray(zf["Gamma0_values"], float)
        J, h = np.asarray(zf["J"], float), float(zf["h"])
        Gam = np.asarray(zf["Gamma"], float)
        sel = np.asarray(zf["selected_branch"])
        ef, eb = np.asarray(zf["forward_energy"], float), np.asarray(zf["backward_energy"], float)
        czz_dmrg = np.asarray(zf["Czz"], float)
    e_dmrg = np.where(np.char.startswith(sel.astype(str), "b"), eb, ef)
    z, diag, flips = operators(L, J, h)
    D = sparse.diags(diag)
    rng = np.random.default_rng(L * 100003 + int(Path(scan).name.split("seed")[1].split("_")[0]))
    E0, E1 = np.zeros(g.size), np.zeros(g.size)
    czz = np.zeros((g.size, L, L))
    for k in range(g.size):
        if np.all(Gam[k] == 0):
            m = np.isclose(diag, diag.min(), atol=1e-12)
            p = m / m.sum()
            E0[k] = diag.min()
            E1[k] = np.min(diag[~m]) if m.sum() == 1 else diag.min()
        else:
            Hk = D - sum(Gam[k, i] * flips[i] for i in range(L))
            w, v = eigsh(Hk, k=2, which="SA", v0=rng.standard_normal(diag.size), tol=TOL)
            o = np.argsort(w)
            E0[k], E1[k] = w[o]
            p = v[:, o[0]] ** 2
        zf_ = z.astype(float)
        czz[k] = (zf_ * p[:, None]).T @ zf_
    r = L // 2 - 1
    stag = lambda C: (-1) ** r * np.mean([C[i, i + r] for i in range(L - r)])  # noqa: E731
    c_ed = np.array([stag(c) for c in czz])
    c_dm = np.array([stag(c) for c in czz_dmrg])
    return {"Gamma0": g, "E0": E0, "E1": E1, "E_dmrg": e_dmrg, "C_ed": c_ed, "C_dmrg": c_dm,
            "dCzz_max": np.abs(czz - czz_dmrg).max(axis=(1, 2))}


def check():
    """Sign convention against the reference builder."""
    from clean_tfim_tn.ed.random_ising import build_random_ising_hamiltonian
    L, rng = 8, np.random.default_rng(1)
    J, G, h = 1 + rng.random(L - 1), 0.3 * (1 + rng.random(L)), 2.5
    ref = build_random_ising_hamiltonian(L, J, G, h=h, bc="open")
    _, diag, flips = operators(L, J, h)
    mine = sparse.diags(diag) - sum(G[i] * flips[i] for i in range(L))
    return float(abs(ref - mine).max())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("max |H_ref - H| =", check(), flush=True)
    from analyze_pbc_reentrant_selected_seed_topology import select_physical_seeds
    summary = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for L in SIZES:
            d = dict(np.load(FIGDIR / "data_raw" / f"reentrant_L{L}.npz"))
            f = OUT / f"ed_L{L}.npz"
            if not f.exists():
                dirs = scan_dirs(L)
                t0 = time.time()
                rows = list(pool.map(realization, [(L, str(dirs[int(s)])) for s in d["seeds"]]))
                print(f"L{L}: {len(rows)} realizations, {time.time() - t0:.0f} s", flush=True)
                np.savez_compressed(f, seeds=d["seeds"], **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
            z = np.load(f)
            g = d["Gamma0"]
            step = float(g[1] - g[0])
            iR = list(d["odd_r"]).index(int(d["R"]))
            ee = np.array([[c[0] for c in cm.entry_exit(g, c)] for c in z["C_ed"]])
            de = z["E_dmrg"] - z["E0"]
            gap = z["E1"] - z["E0"]
            _, idx = select_physical_seeds(g, z["seeds"], z["C_ed"], **SELECTION)
            sel = np.zeros(z["seeds"].size, bool)
            sel[np.asarray(idx, int)] = True
            dent, dexit = np.abs(ee[:, 0] - d["entry"]), np.abs(ee[:, 1] - d["exit"])
            bad = np.flatnonzero((dent > step) | (dexit > step) | ~np.isfinite(dent) | ~np.isfinite(dexit))
            summary[f"L{L}"] = {
                "n": int(d["seeds"].size), "step": step,
                "check_C_dmrg_vs_figure": float(np.nanmax(np.abs(z["C_dmrg"] - d["curves_r"][:, iR]))),
                "dE_min": float(de.min()), "dE_max": float(de.max()), "dE_median": float(np.median(de)),
                "n_points_dE_gt_1e-6": int((de > 1e-6).sum()), "n_points": int(de.size),
                "gap_min_at_entry_median": float(np.median([gap[n, max(0, int(np.searchsorted(g, d["entry"][n])) - 1):
                                                               int(np.searchsorted(g, d["entry"][n])) + 1].min()
                                                           for n in range(gap.shape[0])])),
                "dC_r_max": float(np.abs(z["C_ed"] - z["C_dmrg"]).max()),
                "dC_r_median_of_max": float(np.median(np.abs(z["C_ed"] - z["C_dmrg"]).max(1))),
                "dCzz_max": float(z["dCzz_max"].max()),
                "n_realizations_dCzz_gt_1e-3": int((z["dCzz_max"].max(1) > 1e-3).sum()),
                "entry_abs_diff_max": float(np.nanmax(dent)), "exit_abs_diff_max": float(np.nanmax(dexit)),
                "entry_abs_diff_median": float(np.nanmedian(dent)), "exit_abs_diff_median": float(np.nanmedian(dexit)),
                "n_entry_or_exit_off_by_more_than_one_step": int(bad.size),
                "seeds_off": d["seeds"][bad].astype(int).tolist(),
                "n_still_reentrant_ED": int(sel.sum()),
                "seeds_not_reentrant_ED": d["seeds"][~sel].astype(int).tolist()}
            print(f"L{L}:", json.dumps(summary[f"L{L}"]), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
