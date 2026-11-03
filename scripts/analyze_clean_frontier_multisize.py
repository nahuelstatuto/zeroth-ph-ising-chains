"""Audit the clean OBC frontier using all compatible completed scans.

This analysis is deliberately physical and topology-blind.  It merges scans
only when L, h and chi agree, locates the maximum slope of the bulk staggered
order parameter, and reports energy-curvature and entropy diagnostics.  The
finite-OBC classical endpoint h*=2(1-1/L) is used to distinguish boundary
cuts from paramagnetic controls.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "clean_ising_gamma_campaign"
OUTPUT = BASE / "clean-frontier-multisize-analysis-v1" / "analysis"
PAPER = {
    0.2: 0.992, 0.4: 0.970, 0.6: 0.931, 0.8: 0.878,
    1.0: 0.802, 1.2: 0.712, 1.4: 0.599, 1.6: 0.453,
    1.8: 0.278, 1.9: 0.154, 1.95: 0.077,
}
CAMPAIGNS = (
    "clean-fig1-L16-wide010-c64-worker1-v1",
    "clean-fig1-L16-wide010-c64-worker2-v1",
    "clean-fig1-L16-wide010-c64-turing-v1",
    "clean-fig1-L16-lowerext010-c64-worker2-v1",
    "clean-fig1-L16-upperext010-c64-worker2-v1",
    "clean-fig1-L32-wide010-c128-worker2-a-v1",
    "clean-fig1-L32-wide010-c128-worker2-b-v1",
    "clean-fig1-L32-wide010-c128-worker2-c-v1",
    "clean-fig1-L32-wide010-c128-worker2-d-v1",
    "clean-fig1-bound-L64-c128-v1",
    "clean-fig1-refine-L64-c128-v1",
    "clean-fig1-refine2-L64-c128-v1",
    "clean-fig1-L64-h08-lower-c128-worker1-v1",
    "clean-fig1-L64-h10-h12-upper-c128-worker2-v1",
    "clean-fig1-L32-followup-c128-turing-v1",
    "clean-fig1-L32-h19-c256-turing-v1",
    "clean-L32-fullgamma-h07-c128-turing-v1",
    "clean-L32-fullgamma-h11-c128-turing-v1",
    "clean-L32-fullgamma-h15-c128-turing-v1",
    "clean-h195-fine-L64-c512-v1",
    "clean-h195-leftfine-L64-c512-turing-v1",
)


def save_figure(fig, path: Path, **kwargs) -> None:
    """Save through a file object so long Windows OneDrive paths work."""
    target = str(path.resolve())
    if os.name == "nt" and not target.startswith("\\\\?\\"):
        target = "\\\\?\\" + target
    with open(target, "wb") as handle:
        fig.savefig(handle, format=path.suffix.lstrip("."), **kwargs)


def second_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    dl = np.diff(y)[:-1] / np.diff(x)[:-1]
    dr = np.diff(y)[1:] / np.diff(x)[1:]
    out[1:-1] = 2.0 * (dr - dl) / (x[2:] - x[:-2])
    return out


def vertex(x: np.ndarray, y: np.ndarray, index: int) -> tuple[float, bool]:
    if index == 0 or index == x.size - 1:
        return float(x[index]), True
    xx, yy = x[index - 1:index + 2], y[index - 1:index + 2]
    if not np.all(np.isfinite(yy)):
        return float(x[index]), True
    p = np.polyfit(xx, yy, 2)
    if p[0] >= 0:
        return float(x[index]), True
    value = float(-p[1] / (2 * p[0]))
    if value < xx[0] or value > xx[-1]:
        return float(x[index]), True
    return value, False


def scan_path(campaign: Path, job: dict) -> Path:
    candidate = Path(job.get("source_directory", ""))
    if candidate.is_dir():
        return candidate
    L, h = int(job["L"]), float(job["h"])
    chi = json.loads((campaign / "campaign_config.json").read_text())["chi_max"]
    return campaign / "scans" / f"L{L}_h_{h:g}_J1_chi{chi}"


def campaign_status_path(campaign: Path) -> Path | None:
    """Prefer the status copy containing the most completed jobs.

    OneDrive can preserve a newer worker status as a conflict copy while the
    canonical campaign_status.json remains at its pre-launch state.  Reading
    the scientifically most complete status avoids dropping valid scans; no
    status file is modified here.
    """
    candidates = sorted(campaign.glob("campaign_status*.json"))
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, float]:
        status = json.loads(path.read_text())
        complete = sum(job.get("status") == "complete" for job in status.get("jobs", []))
        return complete, path.stat().st_mtime

    return max(candidates, key=score)


def load_groups() -> tuple[dict, list[str]]:
    groups: dict[tuple[int, float, int], list[dict]] = defaultdict(list)
    sources: list[str] = []
    for name in CAMPAIGNS:
        campaign = BASE / name
        status_path = campaign_status_path(campaign)
        if status_path is None:
            continue
        config = json.loads((campaign / "campaign_config.json").read_text())
        chi = int(config["chi_max"])
        status = json.loads(status_path.read_text())
        for job in status.get("jobs", []):
            if job.get("status") != "complete":
                continue
            path = scan_path(campaign, job)
            archive = path / "continuation_summary.npz"
            if not archive.is_file():
                continue
            with np.load(archive, allow_pickle=False) as z:
                groups[(int(job["L"]), float(job["h"]), chi)].append({
                    "gamma": np.asarray(z["Gamma_values"], float),
                    "cstag": np.asarray(z["Cstag"], float),
                    "energy": np.asarray(z["selected_energy_density"], float),
                    "entropy": np.asarray(z["half_chain_entropy"], float),
                    "bond": np.asarray(z["selected_max_bond_dimension"], int),
                    "branch": np.asarray(z["branch_disagreement"], bool),
                    "energy_diff": np.asarray(z["absolute_energy_difference"], float),
                    "fidelity": np.asarray(z["branch_fidelity"], float),
                    "source": str(archive.resolve()),
                })
            sources.append(str(archive.resolve()))
    return groups, sources


def merge(scans: list[dict]) -> dict[str, np.ndarray]:
    points: dict[float, dict] = {}
    duplicate_conflicts = 0
    for scan in scans:
        for i, gamma in enumerate(scan["gamma"]):
            key = round(float(gamma), 12)
            point = {k: scan[k][i] for k in (
                "cstag", "energy", "entropy", "bond", "branch",
                "energy_diff", "fidelity",
            )}
            if key in points:
                if not np.isclose(points[key]["energy"], point["energy"], rtol=1e-9, atol=1e-11):
                    # Independent continuations can retain different states
                    # close to an avoided crossing.  Keep the variationally
                    # lower-energy state and never average their observables.
                    duplicate_conflicts += 1
                    if point["energy"] < points[key]["energy"]:
                        points[key] = point
                continue
            points[key] = point
    keys = sorted(points)
    return {
        "gamma": np.asarray(keys),
        **{name: np.asarray([points[key][name] for key in keys]) for name in points[keys[0]]},
        "duplicate_conflict_count": duplicate_conflicts,
    }


def analyze(key: tuple[int, float, int], scans: list[dict]) -> tuple[dict, dict]:
    L, h, chi = key
    data = merge(scans)
    gamma = data["gamma"]
    start, stop = L // 4, 3 * L // 4
    bulk = np.mean(data["cstag"][:, start:stop, start:stop], axis=(1, 2))
    mid = 0.5 * (gamma[:-1] + gamma[1:])
    response = -np.diff(bulk) / np.diff(gamma)
    peak_index = int(np.nanargmax(response))
    peak_gamma, peak_fit_limited = vertex(mid, response, peak_index)
    susceptibility = -second_derivative(gamma, data["energy"])
    susceptibility_index = int(np.nanargmax(susceptibility))
    susceptibility_gamma, susceptibility_fit_limited = vertex(
        gamma, susceptibility, susceptibility_index
    )
    entropy_index = int(np.nanargmax(data["entropy"]))
    entropy_gamma, entropy_fit_limited = vertex(gamma, data["entropy"], entropy_index)
    endpoint = 2.0 * (1.0 - 1.0 / L)
    row = {
        "L": L, "chi": chi, "h": h, "h_classical_endpoint": endpoint,
        "is_boundary_cut": h < endpoint - 1e-12,
        "paper_Gamma": PAPER.get(round(h, 2), np.nan),
        "point_count": int(gamma.size), "Gamma_min": float(gamma[0]),
        "Gamma_max": float(gamma[-1]),
        "bulk_response_Gamma": peak_gamma,
        "bulk_response_sample_Gamma": float(mid[peak_index]),
        "bulk_response_value": float(response[peak_index]),
        "bulk_response_edge_limited": bool(peak_index in (0, response.size - 1)),
        "bulk_response_fit_limited": peak_fit_limited,
        "susceptibility_Gamma": susceptibility_gamma,
        "susceptibility_sample_Gamma": float(gamma[susceptibility_index]),
        "susceptibility_edge_limited": bool(susceptibility_index in (1, gamma.size - 2)),
        "susceptibility_fit_limited": susceptibility_fit_limited,
        "entropy_Gamma": entropy_gamma,
        "entropy_edge_limited": bool(entropy_index in (0, gamma.size - 1)),
        "entropy_fit_limited": entropy_fit_limited,
        "branch_disagreement_count": int(np.sum(data["branch"])),
        "bond_saturation_count": int(np.sum(data["bond"] >= chi)),
        "max_energy_difference": float(np.max(data["energy_diff"])),
        "max_branch_infidelity": float(np.max(np.maximum(0.0, 1.0 - data["fidelity"]))),
        "duplicate_conflict_count": int(data["duplicate_conflict_count"]),
    }
    curve = {"gamma": gamma, "bulk": bulk, "response_mid": mid, "response": response}
    return row, curve


def main() -> None:
    import matplotlib.pyplot as plt

    groups, sources = load_groups()
    rows, curves = [], {}
    for key in sorted(groups):
        row, curve = analyze(key, groups[key])
        rows.append(row)
        curves[key] = curve
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "clean_frontier_multisize.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    boundary = [r for r in rows if r["is_boundary_cut"]]
    missing = [r for r in boundary if r["bulk_response_edge_limited"]]
    qc = [r for r in rows if r["branch_disagreement_count"] or r["bond_saturation_count"]]
    summary = {
        "rows": rows,
        "boundary_cut_count": len(boundary),
        "paramagnetic_control_count": len(rows) - len(boundary),
        "bulk_response_edge_limited": [{"L": r["L"], "h": r["h"], "chi": r["chi"]} for r in missing],
        "qc_flagged": [{"L": r["L"], "h": r["h"], "chi": r["chi"], "branches": r["branch_disagreement_count"], "saturations": r["bond_saturation_count"]} for r in qc],
        "source_files": sources,
        "guardrails": [
            "All estimates are finite-size OBC pseudocritical responses.",
            "Cuts above h*=2(1-1/L) are paramagnetic controls, not missing boundary points.",
            "Different chi values are not merged.",
        ],
    }
    (OUTPUT / "clean_frontier_multisize.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)
    ph = np.asarray(sorted(PAPER))
    axes[0].plot(ph, [PAPER[x] for x in ph], "k-", lw=1.7, label="paper (digitalized)")
    axes[0].plot([0, 2], [1, 0], alpha=0)
    colors = {16: "#0072B2", 32: "#D55E00", 64: "#009E73"}
    for L in (16, 32, 64):
        chosen = [r for r in rows if r["L"] == L and r["is_boundary_cut"]]
        # Prefer the highest chi if a cut occurs in more than one campaign family.
        best = {}
        for r in chosen:
            if r["h"] not in best or r["chi"] > best[r["h"]]["chi"]:
                best[r["h"]] = r
        chosen = [best[h] for h in sorted(best)]
        axes[0].plot([r["h"] for r in chosen], [r["bulk_response_Gamma"] for r in chosen], "o-", color=colors[L], label=f"L={L}")
        axes[1].plot([r["h"] for r in chosen], [r["bulk_response_Gamma"] - r["paper_Gamma"] for r in chosen], "o-", color=colors[L], label=f"L={L}")
    axes[0].set(xlabel=r"$h$", ylabel=r"$\Gamma_\mathrm{peak}$", title="Clean OBC finite-size frontier")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set(xlabel=r"$h$", ylabel=r"$\Gamma_\mathrm{peak}-\Gamma_\mathrm{paper}$", title="Finite-size displacement")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(frameon=False)
    save_figure(fig, OUTPUT / "clean_frontier_multisize.png", dpi=220)
    save_figure(fig, OUTPUT / "clean_frontier_multisize.pdf")
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ("boundary_cut_count", "paramagnetic_control_count", "bulk_response_edge_limited", "qc_flagged")}, indent=2))


if __name__ == "__main__":
    main()
