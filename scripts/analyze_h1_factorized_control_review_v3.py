"""B8 (review v3, R2-MC4): does the first H1 cycle of the odd open chains survive when the
correlation matrix is replaced by the product of local magnetizations?

For every odd-L cut of Fig. 6 (clean OBC, L = 31 and 63, chi = 128, the cuts stored in
paper_figures/fig_clean_frontier/data_raw/clean_frontier_oddL_compact.npz) we
  * read <sigma^z_i> on every site from the selected MPS saved with the scan (selected_mps.h5);
  * build C^prod_ij = (-1)^|i-j| m_i m_j for i != j, with unit diagonal;
  * compute the H1 intervals of the Vietoris-Rips filtration of D = clip(1 - C, 0, 2) exactly as
    prepare_data.h1_diagram, and the onset with the rule of the figure (first QC-valid interval
    where N1 = #lifetimes > tau = 0.04 rises; QC flag = branch disagreement);
  * check that the same code on the stored Cstag reproduces the onsets of the figure summary.
Also records max |Cstag - C^prod| off the diagonal and for the end-end and end-centre pairs.
Run with the tn_tda environment (TeNPy). Output: results/review_v3/h1-factorized-control-v1/
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper_figures/fig_clean_frontier"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(FIG))
from prepare_data import h1_diagram  # noqa: E402
from clean_tfim_tn.io.clean_ising_mps_hdf5 import load_clean_ising_mps_collection  # noqa: E402

OUT = ROOT / "results/review_v3/h1-factorized-control-v1"
TAU = 0.04


def first_rise(g, diagrams, flag, tau=TAU):
    n1 = np.array([int(np.sum(d[:, 1] - d[:, 0] > tau)) if len(d) else 0 for d in diagrams])
    rise = (np.diff(n1) >= 1) & ~(flag[:-1] | flag[1:])
    if not rise.any():
        return n1, None
    i = int(np.flatnonzero(rise)[0])
    return n1, float(0.5 * (g[i] + g[i + 1]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(FIG / "data_raw/clean_frontier_oddL_compact.npz")
    index = json.loads(str(z["index_json"]))
    figsum = json.loads((FIG / "output/fig_clean_frontier_summary.json").read_text(encoding="utf-8"))
    fig_onset = {(c["L"], c["h"]): (c["H1"]["x"] if c["H1"] else None, c["reference"]) for c in figsum["odd_cuts"]}
    rows, arrays = [], {}
    for item in index:
        L, h, p = item["L"], item["h"], item["prefix"]
        src = ROOT / item["sources"][0]
        with np.load(src, allow_pickle=False) as s:
            g = np.asarray(s["Gamma_values"], float)
            C = np.asarray(s["Cstag"], float)
            flag = np.asarray(s["branch_disagreement"], bool)
        states = load_clean_ising_mps_collection(src.parent / "selected_mps.h5")["selected_states"]
        keys = sorted(states)
        mz = np.array([np.real(states[k].expectation_value("Sigmaz")) for k in keys])
        assert mz.shape == (g.size, L)
        assert np.allclose(mz[:, 0], z[f"{p}_mz_end_first"]) and np.allclose(mz[:, L // 2], z[f"{p}_mz_centre"])
        sign = (-1.0) ** np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
        Cp = sign[None] * mz[:, :, None] * mz[:, None, :]
        Cp[:, np.arange(L), np.arange(L)] = 1.0
        off = ~np.eye(L, dtype=bool)
        dev = np.abs(C - Cp)[:, off].max(axis=1)
        d_true = [h1_diagram(m) for m in C]
        d_prod = [h1_diagram(m) for m in Cp]
        n_true, on_true = first_rise(g, d_true, flag)
        n_prod, on_prod = first_rise(g, d_prod, flag)
        life = lambda ds: np.array([float((d[:, 1] - d[:, 0]).max()) if len(d) else 0.0 for d in ds])  # noqa: E731
        lt, lp = life(d_true), life(d_prod)
        fo, ref = fig_onset.get((L, h), (None, None))
        rows.append({"L": L, "h": h, "reference": ref, "onset_figure": fo, "onset_true_recomputed": on_true,
                     "onset_product": on_prod, "max_life_true": float(lt.max()), "max_life_product": float(lp.max()),
                     "max_offdiag_dev": float(dev.max()),
                     "max_dev_end_end": float(np.abs(C[:, 0, L - 1] - Cp[:, 0, L - 1]).max()),
                     "max_dev_end_centre": float(np.abs(C[:, 0, L // 2] - Cp[:, 0, L // 2]).max())})
        arrays.update({f"{p}_Gamma": g, f"{p}_mz": mz, f"{p}_N1_true": n_true, f"{p}_N1_product": n_prod,
                       f"{p}_maxlife_true": lt, f"{p}_maxlife_product": lp, f"{p}_offdiag_dev": dev})
        print(json.dumps(rows[-1]), flush=True)
    np.savez_compressed(OUT / "per_cut.npz", **arrays)
    (OUT / "summary.json").write_text(json.dumps({"tau": TAU, "cuts": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
