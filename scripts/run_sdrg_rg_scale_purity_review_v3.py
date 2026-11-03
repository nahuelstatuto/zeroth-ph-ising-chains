"""Review v3 (R1 objection 2, R2 major comment 3, triage B2): purity of the H0 hierarchy
stratified by the RG scale at which each SDRG cluster forms, and SDRG partitions stopped at a
finite RG scale.

Question. The SDRG is asymptotically exact at the infinite-randomness fixed point, but at fixed
delta the global purity of C1 (run_sdrg_disorder_strength_review_v3) does not grow with L. Are
the late clusters, formed at low energy where the effective disorder is strong, purer than the
early ones, and is the L dependence a mixture of strata?

For every realization the SDRG flow of compare_sdrg_h0_clusters_review_v1.sdrg_partition is
rerun with bookkeeping (the labels are checked to be identical). For every final cluster
(sites decimated together by one field) it stores:
  ln_omega_dec    ln of the field that decimates it (the RG scale Omega at its decimation)
  rho_dec         active clusters / L just before its decimation (the RG stage, 1 -> 0)
  ln_omega_first  ln Omega of the first bond decimation in its history (nan for single sites)
  ln_omega_last   ln Omega of the last bond decimation
  n_bond          number of bond decimations that built it
  size, contiguous
and its purity in the H0, average, NN trees (purity_variants of compare_sdrg_linkages_review_v2:
pair purity per cluster, so pair-weighted sums reproduce the global purity of the paper).
In the fixed_logmean family every ln coupling is delta times a delta-independent number, so
the flow, the strata and the clusters are the same for every delta; only the ground state
changes. rho_dec is also delta-independent and comparable across L.

Pair stratification. A large cluster is built by bond decimations at many scales, so its
decimation scale mixes early and late merges. Every intra-cluster pair (i, j) is therefore also
assigned the stage at which the SDRG joined it (the bond decimation that merged the parts
containing i and j), and its pair purity in each tree (the fraction of its cluster in the
smallest subtree containing both, the summand of the paper's purity) is summed into octave bins
of that stage, RHO_EDGES. Stored per realization: pairbin_den (pairs) and pairbin_num_{tree}.

B2, stopped flow. The partition when the number of active clusters first drops to rho_stop L
(decimated clusters plus every active cluster as one label) is scored with the same trees:
pair-weighted purity over all clusters and over the active ones only, and their baseline
(the stopped partition of the previous realization on the H0 tree).

Ensembles:
  fixed_logmean  power-law family of C1, delta in DELTAS, L = 32-256, the seeds of C1
                 (seeds[1:301] of results/review_v1/free-fermion-mst-clusters-v2, previous
                 seed as baseline), so the global purities must reproduce C1.
  box            the fixed-mean box ensemble of the paper (Sec. IV B), L = 32-512,
                 seeds[0:300] with a circular baseline (300 realizations at every L).
OBC, h = 0, Gamma0 = 1, exact free fermions, raw C = Cstag, D = 1 - C.

Run (Anaconda Prompt, env tn_tda):
  python scripts/run_sdrg_rg_scale_purity_review_v3.py --workers 12
Output: results/review_v3/sdrg-rg-scale-v1/{tag}.npz (+ summary.json). Each block is saved on
completion and skipped on restart.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from clean_tfim_tn.disorder.reentrant import generate_disorder_base  # noqa: E402
from clean_tfim_tn.exact.free_fermion_h0 import cstag_elimination, majorana_state  # noqa: E402
from compare_sdrg_h0_clusters_review_v1 import sdrg_partition  # noqa: E402
from compare_sdrg_h0_hierarchy_review_v1 import dendrogram  # noqa: E402
from compare_sdrg_linkages_review_v2 import contiguous, linkage_tree, purity_variants  # noqa: E402
from run_sdrg_disorder_strength_review_v3 import couplings  # noqa: E402

DATA = ROOT / "results" / "review_v1" / "free-fermion-mst-clusters-v2"
OUT = ROOT / "results" / "review_v3" / "sdrg-rg-scale-v1"
DELTAS = (0.25, 0.5, 1.0, 2.0)
SIZES_LOGMEAN = (32, 64, 128, 256)
SIZES_BOX = (32, 64, 128, 256, 512)
RHO_STOPS = (0.5, 0.25, 0.125, 0.0625)
N_REAL = 300
CUTOFF = 1e-15
TREES = ("H0", "average", "NN")
RHO_EDGES = np.concatenate([[0.0], 2.0 ** -np.arange(0, 10)[::-1]])   # (0, 2^-9], ..., (1/2, 1]
CL_KEYS = ("ln_omega_dec", "rho_dec", "ln_omega_first", "ln_omega_last", "n_bond", "size", "contiguous")


def sdrg_flow(J, h, rho_stops=RHO_STOPS):
    """sdrg_partition with the RG scale of every cluster and the partitions of the stopped flow."""
    fields = list(map(float, h))
    bonds = list(map(float, J))
    L = len(fields)
    members = [[i] for i in range(L)]
    first, last, nb = [np.nan] * L, [np.nan] * L, [0] * L
    labels = np.full(L, -1)
    join = np.full((L, L), np.nan)                    # stage rho at which the SDRG joined i and j
    info = []
    stops = sorted(rho_stops, reverse=True)
    stopped = {}
    nxt = 0
    while fields:
        n = len(fields)
        while stops and n <= stops[0] * L:
            lab_s, k = labels.copy(), nxt
            for m in members:
                lab_s[m] = k
                k += 1
            stopped[stops.pop(0)] = (lab_s, nxt, float(np.log(max(max(fields), max(bonds, default=0.0)))))
        kf = int(np.argmax(fields))
        kb = int(np.argmax(bonds)) if bonds else -1
        if kb >= 0 and bonds[kb] > fields[kf]:
            lo = float(np.log(bonds[kb]))
            X, Y = members[kb], members[kb + 1]
            join[np.ix_(X, Y)] = n / L
            join[np.ix_(Y, X)] = n / L
            first[kb] = np.nanmax([first[kb], first[kb + 1], lo])     # earliest = largest Omega
            last[kb] = lo
            nb[kb] = nb[kb] + nb[kb + 1] + 1
            fields[kb] = fields[kb] * fields[kb + 1] / bonds[kb]
            members[kb] = members[kb] + members[kb + 1]
            del fields[kb + 1], members[kb + 1], bonds[kb], first[kb + 1], last[kb + 1], nb[kb + 1]
        else:
            labels[members[kf]] = nxt
            idx = np.asarray(members[kf])
            info.append((float(np.log(fields[kf])), n / L, first[kf], last[kf], nb[kf], idx.size,
                         bool(contiguous(idx))))
            nxt += 1
            if 0 < kf < n - 1:
                bonds[kf - 1] = bonds[kf - 1] * bonds[kf] / fields[kf]
                del bonds[kf]
            elif kf == 0 and n > 1:
                del bonds[0]
            elif kf == n - 1 and n > 1:
                del bonds[kf - 1]
            del fields[kf], members[kf], first[kf], last[kf], nb[kf]
    for r in stops:                                   # stop never reached: the full flow
        stopped[r] = (labels.copy(), nxt, np.nan)
    return labels, info, stopped, join


def pair_bins(lab, merges, join):
    """Sum of pair purities and number of pairs per bin of the SDRG joining stage."""
    pp = np.full(join.shape, np.nan)
    for A, B in merges:
        A, B = np.asarray(A), np.asarray(B)
        la, lb = lab[A], lab[B]
        n = A.size + B.size
        for k in set(la.tolist()) & set(lb.tolist()):
            ia, ib = A[la == k], B[lb == k]
            pp[np.ix_(ia, ib)] = (ia.size + ib.size) / n
    iu = np.triu_indices(lab.size, 1)
    pv = np.where(np.isnan(pp[iu]), pp.T[iu], pp[iu])
    jv = join[iu]
    ok = np.isfinite(jv)
    assert np.all(np.isfinite(pv[ok])), "intra-cluster pair without purity"
    b = np.clip(np.searchsorted(RHO_EDGES, jv[ok], side="left") - 1, 0, RHO_EDGES.size - 2)
    nb = RHO_EDGES.size - 1
    return np.bincount(b, weights=pv[ok], minlength=nb), np.bincount(b, minlength=nb).astype(float)


def pair_purity(per, lab, keep=None):
    sz = Counter(lab.tolist())
    ks = [k for k in per if keep is None or keep(k)]
    w = np.array([sz[k] * (sz[k] - 1) / 2 for k in ks], float)
    return float(np.dot([per[k] for k in ks], w) / w.sum()) if w.sum() > 0 else np.nan


def get_couplings(L, seed, ens, delta):
    if ens == "box":
        r = generate_disorder_base(L, seed).realize(1.0)
        return np.asarray(r.J, float), np.asarray(r.Gamma, float)
    return couplings(L, seed, delta, ens)


def task(args):
    L, seed, seed_prev, delta, ens = args
    J, G = get_couplings(L, seed, ens, delta)
    lab, info, stopped, join = sdrg_flow(J, G)
    assert np.array_equal(lab, np.asarray(sdrg_partition(J, G))), "bookkeeping changed the SDRG partition"
    _, _, stopped_prev, _ = sdrg_flow(*get_couplings(L, seed_prev, ens, delta))
    W, S, _ = majorana_state(J, G)
    C = cstag_elimination(W, cutoff=CUTOFF)
    iu = np.triu_indices(L, 1)
    same = lab[iu[0]] == lab[iu[1]]
    qc = {"cmax_excess": float(C[iu].max() - 1.0), "cmin": float(C[iu].min()),
          "intra_tie_frac": float(np.mean(np.abs(C[iu][same]) < CUTOFF)) if same.any() else 0.0,
          "smin": float(S.min()), "ln_omega0": float(np.log(max(J.max(), G.max())))}
    D = np.clip(1.0 - C, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    order = np.argsort(D[iu], kind="stable")
    merges = {"H0": dendrogram(L, list(zip(iu[0][order], iu[1][order])))[1],
              "average": linkage_tree(L, D, "average")[1],
              "NN": dendrogram(L, [(i, i + 1) for i in np.argsort(np.diagonal(D, 1), kind="stable")])[1]}
    # per final cluster: scale and purity in every tree (nan for single sites)
    per = {m: purity_variants(lab, merges[m]) for m in TREES}
    cl = {k: np.array([row[i] for row in info], float) for i, k in enumerate(CL_KEYS)}
    for m in TREES:
        cl[f"pur_{m}"] = np.array([per[m].get(k, np.nan) for k in range(len(info))], float)
    glob = {m: pair_purity(per[m], lab) for m in TREES}
    for m in TREES:
        num, den = pair_bins(lab, merges[m], join)
        cl[f"pairbin_num_{m}"] = num
        assert abs(num.sum() / den.sum() - glob[m]) < 1e-9, "pair bins do not add up to the global purity"
    cl["pairbin_den"] = den
    # stopped flow (B2)
    stop = {}
    for r in RHO_STOPS:
        lab_s, n_dec, ln_om = stopped[r]
        lab_p, n_dec_p, _ = stopped_prev[r]
        for m in TREES:
            p = purity_variants(lab_s, merges[m])
            stop[f"stop{r:g}_{m}_all"] = pair_purity(p, lab_s)
            stop[f"stop{r:g}_{m}_active"] = pair_purity(p, lab_s, keep=lambda k: k >= n_dec)
        pb = purity_variants(lab_p, merges["H0"])
        stop[f"stop{r:g}_baseline_all"] = pair_purity(pb, lab_p)
        stop[f"stop{r:g}_baseline_active"] = pair_purity(pb, lab_p, keep=lambda k: k >= n_dec_p)
        stop[f"stop{r:g}_ln_omega"] = ln_om
        sz = Counter(lab_s.tolist())
        stop[f"stop{r:g}_frac_sites_ge2"] = sum(n for n in sz.values() if n >= 2) / L
    return glob, cl, stop, qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-real", type=int, default=N_REAL)
    ap.add_argument("--ensembles", nargs="+", default=["fixed_logmean", "box"])
    ap.add_argument("--deltas", type=float, nargs="+", default=list(DELTAS))
    ap.add_argument("--sizes", type=int, nargs="+", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    blocks = []
    for ens in a.ensembles:
        sizes = a.sizes or (SIZES_BOX if ens == "box" else SIZES_LOGMEAN)
        for L in sizes:
            for delta in ([None] if ens == "box" else a.deltas):
                blocks.append((ens, L, delta))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for ens, L, delta in blocks:
            tag = f"box_L{L}" if ens == "box" else f"{ens}_d{delta:g}_L{L}"
            f = out / f"{tag}.npz"
            if f.exists():
                print(f"SKIP {tag}", flush=True)
                continue
            seeds = np.load(DATA / f"random_L{L}.npz")["seeds"].astype(int)
            if ens == "box":
                s = seeds[:a.n_real]
                jobs = [(L, int(s[i]), int(s[i - 1]), None, ens) for i in range(s.size)]   # circular
            else:
                s = seeds[:a.n_real + 1]
                jobs = [(L, int(s[i]), int(s[i - 1]), delta, ens) for i in range(1, s.size)]
            t0 = time.time()
            rows = list(pool.map(task, jobs, chunksize=2))
            arr = {f"{m}_purity": np.array([r[0][m] for r in rows], float) for m in TREES}
            arr.update({k: np.array([r[2][k] for r in rows], float) for k in rows[0][2]})
            arr.update({k: np.array([r[3][k] for r in rows], float) for k in rows[0][3]})
            for k in rows[0][1]:
                if k.startswith("pairbin"):
                    arr[k] = np.stack([r[1][k] for r in rows])
                else:
                    arr[f"cl_{k}"] = np.concatenate([r[1][k] for r in rows])
            arr["rho_edges"] = RHO_EDGES
            arr["cl_real"] = np.concatenate([np.full(r[1]["size"].size, i) for i, r in enumerate(rows)])
            arr["seeds"] = np.array([j[1] for j in jobs])
            arr["seeds_prev"] = np.array([j[2] for j in jobs])
            np.savez_compressed(f, **arr)
            big = arr["cl_size"] >= 2
            late = big & (arr["cl_rho_dec"] <= 0.125)
            w = arr["cl_size"] * (arr["cl_size"] - 1) / 2
            s_ = {m: float(np.nanmean(arr[f"{m}_purity"])) for m in TREES}
            s_["H0_late_pairweighted"] = float(np.nansum(arr["cl_pur_H0"][late] * w[late]) / w[late].sum()) if late.any() else None
            den = arr["pairbin_den"].sum(0)
            s_["pair_strata"] = {f"{RHO_EDGES[i]:.4g}-{RHO_EDGES[i + 1]:.4g}": {
                "pairs": float(den[i]), **{m: float(arr[f"pairbin_num_{m}"].sum(0)[i] / den[i]) for m in TREES}}
                for i in range(den.size) if den[i] > 0}
            s_["meta"] = {"ensemble": ens, "delta": delta, "L": L, "n": len(rows), "seconds": time.time() - t0,
                          "n_clusters_ge2": int(big.sum())}
            s_["qc"] = {"cmax_excess_max": float(arr["cmax_excess"].max()), "cmin_min": float(arr["cmin"].min()),
                        "intra_tie_frac_max": float(arr["intra_tie_frac"].max()), "smin_min": float(arr["smin"].min())}
            summary[tag] = s_
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"{tag} {time.time() - t0:.0f} s | P H0 {s_['H0']:.3f} avg {s_['average']:.3f} NN {s_['NN']:.3f} "
                  f"| pairs H0 by stage " + " ".join(f"{k}:{v['H0']:.3f}" for k, v in s_["pair_strata"].items()) + f" | Cmin {s_['qc']['cmin_min']:.1e} "
                  f"ties {s_['qc']['intra_tie_frac_max']:.1e}", flush=True)


if __name__ == "__main__":
    main()
