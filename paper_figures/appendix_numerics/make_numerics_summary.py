"""Numerical details of every dataset in the paper (Appendix A), read from the campaigns.

For each figure dataset it collects, from the campaign_config.json of every scan used, the
DMRG protocol (sweeps, tolerances, mixer, initial states, branch tolerances), the bond
dimension per size, the field grids and the number of realizations, and from each
continuation_summary.npz the quality flags (branch disagreement, bond-dimension saturation).
Read-only. Output: output/numerics_summary.json (numbers for paper_manuscript/tools/make_numbers.py).
"""

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIGS = ROOT / "paper_figures"
RES = ROOT / "results"
PROTOCOL_KEYS = ("two_site_sweeps", "one_site_sweeps", "max_E_err", "max_S_err", "svd_min", "mixer",
                 "branch_energy_tolerance", "branch_infidelity_tolerance", "selection_energy_tolerance")


def config_of(scan_npz):
    camp = Path(scan_npz).resolve().parents[2]
    return json.loads((camp / "campaign_config.json").read_text(encoding="utf-8")), camp.name


def flags(scan_npz):
    with np.load(scan_npz, allow_pickle=False) as z:
        g = np.asarray(z["Gamma_values"] if "Gamma_values" in z.files else z["Gamma0_values"], float)
        br = np.asarray(z["branch_disagreement"], bool)
        if "selected_max_bond_dimension" in z.files:
            bond = np.asarray(z["selected_max_bond_dimension"], int)
        else:   # random scans store both branches only: conservative, larger of the two
            bond = np.maximum(np.asarray(z["forward_max_bond_dimension"], int),
                              np.asarray(z["backward_max_bond_dimension"], int))
    return g, br, bond


def dataset(name, scans):
    """scans: list of (L, path to continuation_summary.npz)."""
    per_L = defaultdict(lambda: {"scans": 0, "points": 0, "branch": 0, "saturated": 0, "chi": set(),
                                 "grid_min": np.inf, "grid_max": -np.inf, "steps": set(), "campaigns": set(),
                                 "initial_states": set(), "seeds": set()})
    protocol = defaultdict(set)
    for L, path in scans:
        cfg, camp = config_of(path)
        for k in PROTOCOL_KEYS:
            protocol[k].add(json.dumps(cfg.get(k)))
        chi = int(cfg["chi_max"])
        g, br, bond = flags(path)
        d = per_L[L]
        d["scans"] += 1
        d["points"] += int(g.size)
        d["branch"] += int(br.sum())
        d["saturated"] += int(np.sum(bond >= chi))
        d["chi"].add(chi)
        d["grid_min"], d["grid_max"] = min(d["grid_min"], float(g.min())), max(d["grid_max"], float(g.max()))
        d["steps"].add(round(float(np.min(np.diff(g))), 6))
        d["campaigns"].add(camp)
        d["initial_states"].add(" ".join(cfg.get("initial_states", [])))
        seed = Path(path).parent.name
        d["seeds"].add(seed)
    out = {}
    for L, d in sorted(per_L.items()):
        out[str(L)] = {"scans": d["scans"], "points": d["points"], "branch_disagreement": d["branch"],
                       "bond_saturated": d["saturated"], "chi": sorted(d["chi"]),
                       "grid": [d["grid_min"], d["grid_max"]], "finest_step": min(d["steps"]),
                       "steps": sorted(d["steps"]), "campaigns": len(d["campaigns"]),
                       "initial_states": sorted(d["initial_states"])}
    return {"name": name, "sizes": out, "protocol": {k: sorted(v) for k, v in protocol.items()}}


def scans_from_dirs(patterns, L_of):
    out = []
    for pat in patterns:
        for p in sorted(glob.glob(str(RES / pat / "scans" / "*" / "continuation_summary.npz"))):
            out.append((L_of(p), p))
    return out


def main():
    data = {}
    # Fig. 2: clean h = 0 (upstream manifest of the frozen figure package)
    m = json.loads((ROOT / "paper_figures" / "figure_clean_h0" / "data_raw" / "source_manifest.json").read_text(encoding="utf-8"))
    data["clean_h0"] = dataset("clean h=0", [(f["L"], ROOT / f["source_path"]) for f in m["files"]])
    # Fig. 3: random h = 0
    Lof = lambda p: int(Path(p).parent.name.split("_")[0][1:])
    data["random_h0"] = dataset("random h=0", scans_from_dirs(
        ["random_ising_gamma_ensemble/random-h0-Gamma-critical-L8-c64-s0-49-v1",
         "random_ising_gamma_ensemble/random-h0-Gamma-critical-L16-c64-s0-49-v1",
         "random_ising_gamma_ensemble/random-h0-Gamma-critical-L32-c128-s0-49-v1",
         "random_ising_gamma_ensemble/random-h0-Gamma-critical-L64-c128-s0-49-v1",
         "random_ising_gamma_ensemble/random-h0-L64-fine625-c128-s*-v1",
         "random_ising_gamma_ensemble/random-h0-L128-fine625-c256-s*-v1"], Lof))
    # Fig. 4: clean frontier (figure manifest: even L and odd L sources)
    m = json.loads((FIGS / "fig_clean_frontier" / "data_raw" / "source_manifest.json").read_text(encoding="utf-8"))
    paths = [ROOT / s["path"] for s in m["sources"] if s["path"].endswith("continuation_summary.npz")
             and "L64_h_1.95" not in s["path"]]   # unconverged control, not used in the figure
    data["clean_frontier"] = dataset("clean h>0", [(Lof(p), p) for p in paths])
    # Fig. 5: reentrant (all seeds of the four cohorts)
    import csv
    even = []
    for L, cohort in ((16, "L16-reentrant-all1000-complete-v1"), (32, "L32-reentrant-h23125-all1000-complete-v1")):
        rows = csv.DictReader((RES / "random_ising_gamma_ensemble" / cohort / "analysis" / "source_manifest.csv").open(encoding="utf-8"))
        even += [(L, ROOT / r["source_npz"]) for r in rows]
    data["reentrant"] = dataset("reentrant", even + scans_from_dirs(
        ["random_ising_gamma_ensemble/L17-reentrant-h2500-oddL-*",
         "random_ising_gamma_ensemble/L33-reentrant-h23125-oddL-*"], Lof))
    # chi convergence check (random h = 0, L = 128, chi 256 vs 512)
    data["chi_qc_random_L128"] = json.loads((RES / "random_ising_gamma_ensemble" / "random-h0-L128-c512-qc-targeted-v1"
                                             / "analysis_matched_chi256_chi512" / "summary.json").read_text(encoding="utf-8"))
    (HERE / "output").mkdir(exist_ok=True)
    (HERE / "output" / "numerics_summary.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    for k, v in data.items():
        if "sizes" in v:
            print(k, json.dumps({L: {kk: s[kk] for kk in ("scans", "points", "branch_disagreement", "bond_saturated", "chi", "grid", "finest_step", "campaigns")}
                                for L, s in v["sizes"].items()}))
            print("   protocol", v["protocol"])


if __name__ == "__main__":
    main()
