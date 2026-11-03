"""Referee 2, round 2 (4.3): accuracy of the correlations from Gaussian elimination without pivoting.

Cstag(i, i + r) = det M[i:i+r, i:i+r] is computed in the paper by elimination without pivoting
(cstag_elimination). Here it is compared with the same minors from LU with partial pivoting
(numpy.linalg.slogdet), at the critical point and in both phases:
  clean chain, g = 0.9, 1.0, 1.1, L = 512 (all start sites, distances r = 1, 2, 4, ..., L - i - 1);
  random chain (fixed mean), Gamma0 = 0.8, 1.0, 1.2, L = 256 and 512, 10 realizations each.
Reported: largest absolute difference, and largest relative difference where C > 1e-8.
Run:  python scripts/check_pivoting_free_fermion_review_v2.py
Output: results/review_v2/pivoting-check-v1/summary.json
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clean_tfim_tn.disorder.reentrant import generate_disorder_base  # noqa: E402
from clean_tfim_tn.exact.free_fermion_h0 import cstag_elimination, majorana_state, string_matrix  # noqa: E402

OUT = ROOT / "results" / "review_v2" / "pivoting-check-v1"


def compare(args):
    kind, L, g, seed = args
    if kind == "clean":
        J, Gam = np.ones(L - 1), np.full(L, g)
    else:
        r = generate_disorder_base(L, seed).realize(g)
        J, Gam = r.J, r.Gamma
    W = majorana_state(J, Gam)[0]
    C = cstag_elimination(W)
    M = string_matrix(W)
    starts = np.unique(np.linspace(0, L - 2, 9).astype(int))
    abs_err, rel_err, n = 0.0, 0.0, 0
    for i in starts:
        rs = sorted({1, 2, 3} | {2 ** k for k in range(1, 10)} | {L - 1 - i})
        for r in rs:
            if r < 1 or i + r > L - 1:
                continue
            sign, logdet = np.linalg.slogdet(M[i:i + r, i:i + r])
            ref = sign * np.exp(logdet)
            abs_err = max(abs_err, abs(C[i, i + r] - ref))
            if ref > 1e-8:
                rel_err = max(rel_err, abs(C[i, i + r] - ref) / ref)
            n += 1
    return kind, L, g, seed, abs_err, rel_err, n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("clean", 512, g, None) for g in (0.9, 1.0, 1.1)]
    jobs += [("random", L, g, s) for L in (256, 512) for g in (0.8, 1.0, 1.2) for s in range(10)]
    with ProcessPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(compare, jobs))
    out = {"pairs": int(sum(r[6] for r in rows)), "max_abs_diff": float(max(r[4] for r in rows)),
           "max_rel_diff_C_gt_1e-8": float(max(r[5] for r in rows)),
           "by_case": [{"kind": r[0], "L": r[1], "g": r[2], "seed": r[3], "max_abs": r[4], "max_rel": r[5], "pairs": r[6]} for r in rows]}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("pairs", "max_abs_diff", "max_rel_diff_C_gt_1e-8")}))


if __name__ == "__main__":
    main()
