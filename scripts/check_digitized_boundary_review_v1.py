"""Referee 1 (4.2): uncertainty of the reference boundary digitized from Fig. 1 of Lajko et al. (2020).

The source image is not available for a pixel-level estimate, so three internal checks are used.
  anchors        the two exact limits, Gamma_c(0) = J (Pfeuty) and the linear law
                 Gamma = a (2J - h) with the hard-dimer slope a = 1.526492 near the multicritical point,
                 compared with the digitized points at h = 0, 1.9 and 1.95
  smoothness     scatter of the digitized points around smooth fits (polynomials of degree 4-6 in h,
                 leave-one-out), an estimate of the random reading error of each point
  interpolation  difference between the linear interpolation used in the paper and a monotone cubic
                 (PCHIP) interpolation of the same points, at the fields of the cuts
Run:  python scripts/check_digitized_boundary_review_v1.py
Output: results/review_v1/digitization-v1/summary.json
"""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "review_v1" / "digitization-v1"
REF_H = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.9, 1.95, 2.0])
REF_G = np.array([1.0, 0.992, 0.970, 0.931, 0.878, 0.802, 0.712, 0.599, 0.453, 0.278, 0.154, 0.077, 0.0])
SLOPE = 1.526492
CUTS = np.round(np.arange(0.1, 1.95, 0.1), 2)      # fields of the cuts in Sec. V (even and odd chains)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    anchors = {"h0": float(REF_G[0] - 1.0)}
    for h in (1.9, 1.95):
        anchors[f"h{h}"] = float(REF_G[REF_H == h][0] - SLOPE * (2.0 - h))
    loo = {}
    for deg in (4, 5, 6):
        res = []
        for k in range(1, len(REF_H) - 1):           # keep the exact end points in every fit
            m = np.arange(len(REF_H)) != k
            p = np.polyfit(REF_H[m], REF_G[m], deg)
            res.append(REF_G[k] - np.polyval(p, REF_H[k]))
        loo[f"deg{deg}"] = {"rms": float(np.sqrt(np.mean(np.square(res)))), "max": float(np.max(np.abs(res)))}
    lin = np.interp(CUTS, REF_H, REF_G)
    cub = PchipInterpolator(REF_H, REF_G)(CUTS)
    d = np.abs(lin - cub)
    out = {"anchors_deviation": anchors, "leave_one_out": loo,
           "interp_linear_vs_pchip": {"cuts": CUTS.tolist(), "abs_diff": d.tolist(), "max": float(d.max()),
                                      "max_h_1.1_1.9": float(d[(CUTS >= 1.1) & (CUTS <= 1.9)].max())}}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
