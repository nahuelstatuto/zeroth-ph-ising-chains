"""Build the compact, hashed input of the clean-frontier figure.

Reads (read-only) every clean OBC frontier scan, merges the scans of each
(L, h, chi) with the rule of scripts/analyze_clean_frontier_multisize.py
(duplicate Gamma: keep the lower-energy state, never average), and stores per
group only what the figure needs:

    Gamma, bulk m_s^2 (window [L/4, 3L/4)), the L-1 finite H0 deaths of
    D = 1 - Cstag (single linkage, Kruskal; exact zeros handled), the
    branch-disagreement flag, the selected maximum bond dimension and chi.

Output: data_raw/clean_frontier_compact.npz and data_raw/source_manifest.json
(SHA-256 of the compact file and of every source continuation_summary.npz).
Run once from the project root:  python paper_figures/fig_clean_frontier/prepare_data.py
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_clean_frontier_multisize as frontier  # noqa: E402

EXTRA_CAMPAIGNS = (
    "clean-L64-H1-active-grid-extension-c128-v1",
    "clean-L64-fullgamma-newh-fine010-c128-v1",
)
FULL_GAMMA_CAMPAIGNS = (
    "clean-L32-fullgamma-h07-c128-turing-v1",
    "clean-L32-fullgamma-h11-c128-turing-v1",
    "clean-L32-fullgamma-h15-c128-turing-v1",
    "clean-L64-fullgamma-newh-fine010-c128-v1",
)
OUT = HERE / "data_raw"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def h0_deaths(C):
    L = C.shape[0]
    D = 1.0 - C
    iu = np.triu_indices(L, 1)
    order = np.argsort(D[iu], kind="stable")
    parent = np.arange(L)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    deaths = []
    for k in order:
        i, j = iu[0][k], iu[1][k]
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            deaths.append(D[i, j])
            if len(deaths) == L - 1:
                break
    return np.sort(np.asarray(deaths, float))


def h1_diagram(C):
    """H1 intervals of the Vietoris-Rips filtration of D = 1 - C (Z2 coefficients).

    Same construction as scripts/analyze_clean_L64_new_topology.py: D is
    symmetrized and clipped to [0, 2]; zero-length intervals are dropped.
    """
    import gudhi

    D = np.clip(1.0 - 0.5 * (C + C.T), 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    tree = gudhi.RipsComplex(distance_matrix=D, max_edge_length=2.0).create_simplex_tree(max_dimension=2)
    tree.persistence(homology_coeff_field=2, min_persistence=-1.0)
    diagram = np.asarray(tree.persistence_intervals_in_dimension(1), float).reshape(-1, 2)
    life = diagram[:, 1] - diagram[:, 0]
    return diagram[np.isfinite(life) & (life > 1e-14)]


def main():
    frontier.CAMPAIGNS = tuple(frontier.CAMPAIGNS) + EXTRA_CAMPAIGNS
    groups, sources = frontier.load_groups()
    full_gamma_sources = {name: str((frontier.BASE / name).resolve()) for name in FULL_GAMMA_CAMPAIGNS}
    arrays, index = {}, []
    for (L, h, chi) in sorted(groups):
        if L not in (16, 32, 64):
            continue
        data = frontier.merge(groups[(L, h, chi)])
        gamma = data["gamma"]
        C = np.asarray(data["cstag"], float)
        a, b = L // 4, 3 * L // 4
        prefix = f"L{L}_h{h:g}_chi{chi}"
        arrays[f"{prefix}_Gamma"] = gamma
        arrays[f"{prefix}_bulk_ms2"] = C[:, a:b, a:b].mean(axis=(1, 2))
        arrays[f"{prefix}_H0_deaths"] = np.stack([h0_deaths(m) for m in C])
        diagrams = [h1_diagram(m) for m in C]
        arrays[f"{prefix}_H1_counts"] = np.array([d.shape[0] for d in diagrams], int)
        arrays[f"{prefix}_H1_intervals"] = (np.concatenate(diagrams) if any(d.size for d in diagrams)
                                           else np.empty((0, 2)))
        arrays[f"{prefix}_branch_disagreement"] = np.asarray(data["branch"], bool)
        arrays[f"{prefix}_max_bond"] = np.asarray(data["bond"], int)
        full = any(full_gamma_sources[n] in s["source"] for n in full_gamma_sources for s in groups[(L, h, chi)])
        index.append({"L": L, "h": h, "chi": chi, "prefix": prefix, "points": int(gamma.size),
                      "Gamma_range": [float(gamma.min()), float(gamma.max())],
                      "includes_full_gamma_scan": bool(full),
                      "duplicate_conflicts": int(data["duplicate_conflict_count"]),
                      "sources": [s["source"] for s in groups[(L, h, chi)]]})
        print(prefix, gamma.size, "full" if full else "", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "clean_frontier_compact.npz"
    np.savez_compressed(target, index_json=np.array(json.dumps(index)), **arrays)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "paper_figures/fig_clean_frontier/prepare_data.py",
        "conventions": "OBC; C=Cstag raw (not connected); topology on D=1-C; Pauli operators",
        "merge_rule": "duplicate Gamma within (L,h,chi): keep lower-energy state, never average",
        "reference_boundary": "13 digitized points of Fig. 1 of Lajko et al., PRB 101, 024203 (2020)",
        "files": [{"file": target.name, "sha256": sha256(target), "size_bytes": target.stat().st_size}],
        "sources": [{"path": str(Path(s).relative_to(ROOT)), "sha256": sha256(s)} for s in sorted(set(sources))],
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("groups:", len(index), "sources:", len(manifest["sources"]))


if __name__ == "__main__":
    main()
