"""Second-order order-by-disorder estimate of the reentrant entry (OBC L16, h=2.5).

For each of the 1000 seeds, enumerate all 2^16 classical configurations of
H_cl = sum_i J_i s_i s_{i+1} - h sum_i s_i, keep the local minima (all single
flip costs > 0) and add the second-order transverse-field shift
E_2 = -Gamma0^2 sum_i gamma_i^2 / dE_i(s), with the seed's own J_i and
transverse-field pattern gamma_i = Gamma_i / Gamma0 (Lajko et al. 2020,
App. C, generalized to random Gamma_i). The predicted entry is the smallest
Gamma0 at which the lowest perturbative configuration has a positive
classical staggered correlation at distance 7. No DMRG, no topology.

Output: reentrant-exit-anatomy-v1/analysis/order_by_disorder_L16.{npz,json}
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "random_ising_gamma_ensemble" / "reentrant-exit-anatomy-v1" / "analysis"
L, H, R = 16, 2.5, 7
G_FINE = np.linspace(0.0, 0.6, 241)


def main():
    z = np.load(OUT / "cache_L16.npz")
    cfg = (((np.arange(2 ** L)[:, None] >> np.arange(L)[None, :]) & 1).astype(np.int8) * 2 - 1)
    stag = cfg * ((-1) ** np.arange(L))[None, :]
    corr_r = (stag[:, :-R] * stag[:, R:]).mean(1)
    pert_entry = np.full(len(z["seeds"]), np.nan)
    any_positive = np.zeros(len(z["seeds"]), bool)
    for k in range(len(z["seeds"])):
        J, gp = z["J"][k], z["gamma_pattern"][k]
        energy = (cfg[:, :-1] * cfg[:, 1:] * J[None, :]).sum(1) - H * cfg.sum(1)
        left = np.zeros(cfg.shape)
        left[:, 1:] = J[None, :] * cfg[:, :-1]
        right = np.zeros(cfg.shape)
        right[:, :-1] = J[None, :] * cfg[:, 1:]
        flip = -2 * cfg * (left + right - H)
        local_min = np.all(flip > 1e-9, axis=1)
        e2 = -(gp[None, :] ** 2 / np.where(flip > 1e-9, flip, np.inf)).sum(1)
        E, E2, cr = energy[local_min], e2[local_min], corr_r[local_min]
        pred = np.array([cr[np.argmin(E + g0 ** 2 * E2)] for g0 in G_FINE])
        any_positive[k] = bool((pred > 0).any())
        up = np.flatnonzero((pred[:-1] <= 0) & (pred[1:] > 0))
        if up.size:
            pert_entry[k] = G_FINE[up[0] + 1]
    sel = np.flatnonzero(z["selected"])
    ok = np.isfinite(pert_entry[sel])
    e, p = z["entry"][sel][ok], pert_entry[sel][ok]
    summary = {
        "selected_with_prediction": int(ok.sum()), "selected": int(sel.size),
        "MAE": float(np.mean(np.abs(p - e))), "constant_median_MAE": float(np.mean(np.abs(np.median(e) - e))),
        "spearman": float(spearmanr(p, e).statistic), "bias": float(np.mean(p - e)),
        "fraction_predicting_positive_C7": {c: float(any_positive[z["sign_class"] == c].mean())
                                            for c in ("raw_reentrant", "single_exit", "always_negative")},
    }
    np.savez(OUT / "order_by_disorder_L16.npz", pert_entry=pert_entry, any_positive=any_positive)
    (OUT / "order_by_disorder_L16.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
