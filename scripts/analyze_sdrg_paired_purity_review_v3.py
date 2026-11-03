"""Referee 2 (major comment 3), review v3: paired difference of dendrogram purity.

Same realizations for both hierarchies (results/review_v2/sdrg-linkages-v1/per_realization.npz,
300 per size, fixed_mean, h = 0, Gamma0 = 1). For each size: mean of H0 - NN and H0 - average
over realizations, 95% percentile bootstrap interval (2000 resamplings) and the fraction of
realizations in which H0 is higher.
Run:  python scripts/analyze_sdrg_paired_purity_review_v3.py
Output: results/review_v3/sdrg-paired-purity-v1/summary.json
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "review_v2" / "sdrg-linkages-v1" / "per_realization.npz"
OUT = ROOT / "results" / "review_v3" / "sdrg-paired-purity-v1"
SIZES = (32, 64, 128, 256, 512)


def main():
    z = np.load(SRC)
    out = {}
    for L in SIZES:
        for other in ("NN", "average"):
            d = z[f"L{L}_H0_purity"] - z[f"L{L}_{other}_purity"]
            d = d[np.isfinite(d)]
            rng = np.random.default_rng(0)
            boot = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(2000)]
            out.setdefault(f"L{L}", {})[f"H0_minus_{other}"] = {
                "mean": float(d.mean()), "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                "fraction_H0_higher": float(np.mean(d > 0)), "fraction_equal": float(np.mean(d == 0)), "n": int(d.size)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
