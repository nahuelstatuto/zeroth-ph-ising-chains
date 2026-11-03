"""Referees 1 (3.3) and 2 (4.1), round 2: effective shift exponent of the clean chain at h = 0.

Reads the exact maxima of Fig. 2 (fig_clean_h0_summary.json of the v2 manuscript) and, with the exact
g_c = 1, computes the local exponents ln[(1 - g*(L)) / (1 - g*(2L))] / ln 2 between consecutive sizes
and power-law fits of 1 - g*(L) over L >= Lmin, for xi_lambda and for the physical response.
Run:  python scripts/analyze_clean_shift_exponent_review_v2.py
Output: results/review_v2/clean-exponent-v1/summary.json
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper_figures" / "fig_clean_h0" / "output" / "fig_clean_h0_summary.json"
OUT = ROOT / "results" / "review_v2" / "clean-exponent-v1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(SRC.read_text(encoding="utf-8"))
    Ls = np.array(sorted(int(k) for k in d["sizes"]))
    out = {"sizes": Ls.tolist(), "grid_step": d["sizes"][str(Ls[-1])]["grid_step"]}
    for key, name in (("landscape_velocity_peak", "xi"), ("physical_peak", "phys")):
        s = np.array([1.0 - d["sizes"][str(L)][key]["x"] for L in Ls])
        loc = [float(np.log(s[i] / s[i + 1]) / np.log(Ls[i + 1] / Ls[i])) for i in range(len(Ls) - 1)]
        fits = {}
        for lmin in (16, 32, 64, 128):
            m = Ls >= lmin
            fits[f"Lmin{lmin}"] = float(-np.polyfit(np.log(Ls[m]), np.log(s[m]), 1)[0])
        out[name] = {"shift": s.tolist(), "local_inverse_nu": loc, "fixed_gc_fits_inverse_nu": fits}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
