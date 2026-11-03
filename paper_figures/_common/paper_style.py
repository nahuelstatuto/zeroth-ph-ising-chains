"""Shared style and estimators for the paper figures in paper_figures.

Every figure imports this module so colors, markers, the peak rule and the
finite-size fit are identical across figures.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SIZE_COLORS = {
    8: "#443983",
    16: "#31688e",
    32: "#35b779",
    64: "#90d743",
    128: "#fde725",
    256: "#d99a00",   # V1: exact free-fermion sizes beyond viridis (checked for CVD separation)
    512: "#8c5a00",
}
PHYSICAL_COLOR = "#0072B2"
TOPOLOGY_COLOR = "#D55E00"
FIT_MIN_SIZE = 16
FIGSIZE_THREE_PANEL = (13.2, 3.8)
WIDTH_RATIOS_THREE_PANEL = (1.0, 1.0, 0.9)
DPI = 300
OUTPUT_FORMATS = ("png", "pdf")

STYLE_FILE = Path(__file__).resolve().parent / "paper.mplstyle"


def apply_style():
    """Load the shared rcParams from paper.mplstyle (the single place to edit the style)."""
    plt.rcdefaults()
    plt.style.use(STYLE_FILE)


def quadratic_peak(x, y):
    """Discrete argmax refined by a local three-point parabola.

    Edge maxima and non-concave windows keep the discrete value.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    i = int(np.argmax(y))
    result = {"x": float(x[i]), "value": float(y[i]), "discrete_x": float(x[i]),
              "refined": False}
    if i == 0 or i == len(y) - 1:
        return result
    c = np.polyfit(x[i - 1:i + 2] - x[i], y[i - 1:i + 2], 2)
    if c[0] >= 0:
        return result
    offset = -c[1] / (2 * c[0])
    if not x[i - 1] - x[i] <= offset <= x[i + 1] - x[i]:
        return result
    result.update({"x": float(x[i] + offset), "value": float(np.polyval(c, offset)),
                   "refined": True})
    return result


def inverse_size_fit(sizes, values, min_size=FIT_MIN_SIZE):
    """Unweighted fit value(L) = g_inf + a/L over L >= min_size."""
    sizes = np.asarray(sizes, float)
    values = np.asarray(values, float)
    mask = sizes >= min_size
    coef, cov = np.polyfit(1.0 / sizes[mask], values[mask], 1, cov=True)
    resid = np.polyval(coef, 1.0 / sizes[mask]) - values[mask]
    return {"g_inf": float(coef[1]), "g_inf_fit_std": float(np.sqrt(cov[1, 1])),
            "slope": float(coef[0]), "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "sizes": [int(v) for v in sizes[mask]]}


def power_law_fit(sizes, values, min_size=FIT_MIN_SIZE):
    """Unweighted log-log fit value(L) = A L^(-p) over L >= min_size."""
    sizes = np.asarray(sizes, float)
    values = np.asarray(values, float)
    mask = sizes >= min_size
    slope, intercept = np.polyfit(np.log(sizes[mask]), np.log(values[mask]), 1)
    return {"A": float(np.exp(intercept)), "p": float(-slope),
            "sizes": [int(v) for v in sizes[mask]]}


def panel_labels(axes, letters="abcdefgh"):
    for ax, letter in zip(axes, letters):
        ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes,
                fontweight="bold", va="top")


def save_figure(fig, output_dir, basename):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in OUTPUT_FORMATS:
        fig.savefig(output_dir / f"{basename}.{ext}", dpi=DPI, bbox_inches="tight")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(data_dir):
    """Check every file listed in data_raw/source_manifest.json by SHA-256."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "source_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if sha256(data_dir / entry["file"]) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {entry['file']}")
    return manifest
