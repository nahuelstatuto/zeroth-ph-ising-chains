# Zeroth persistent homology of ground-state correlations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23915648.svg)](https://doi.org/10.5281/zenodo.23915648)

Code accompanying

> F. Belchi and N. Statuto, *Zeroth persistent homology of ground-state correlations at clean and
> disordered quantum transitions* (submitted to Phys. Rev. B).

The repository contains everything needed to recompute the results of the paper: the simulation code
(exact free fermions, exact diagonalization and DMRG with TeNPy), the frozen specifications of every
simulation campaign, the analysis scripts (persistent homology, landscape velocity, strong-disorder
renormalization group) and the code of every figure. Large simulation outputs (DMRG scans and the saved
matrix-product states) are **not** included, since they can be regenerated with the code. The raw data
of Figs. 2 and 3, which come from the exact free-fermion solution, are included, so those two figures can
be redrawn directly.

## Model and conventions

Antiferromagnetic Ising chain in transverse and longitudinal fields,

    H = sum_b J_b Z_b Z_{b+1} - sum_i Gamma_i X_i - h sum_i Z_i ,

with Pauli operators. Clean chain J_b = J, Gamma_i = Gamma. Random chain with the `fixed_mean` ensemble,
J_b = 1 + u_b and Gamma_i = Gamma_0 (1 + v_i) with u, v uniform in [0, 1], shifted in each realization
so that the means are exactly 1.5 and 1.5 Gamma_0 (`clean_tfim_tn.disorder.reentrant`). Open boundary
conditions unless stated otherwise (the periodic rings of Sec. VI use exact diagonalization).

- Correlations are raw (not connected) staggered correlators, `C = Cstag`, stored as `/data/C`.
- All persistent homology is computed on `D = 1 - C`.

## Installation

Python 3.11 was used. With conda,

    conda env create -f environment.yml
    conda activate tn_tda
    pip install -e .

or with pip only, `pip install -e .` (this installs the package `clean_tfim_tn` from `src/`).

## Layout

| folder | content |
|---|---|
| `src/clean_tfim_tn/` | package: models, exact free fermions (`exact/`), ED (`ed/`), DMRG continuation in the field (`tn/`), observables, HDF5/NPZ I/O |
| `scripts/` | the scripts the paper depends on (`RELEASE_FILES.json` lists them), run from the repository root |
| `campaign_specs/` | frozen specifications of the simulation campaigns (sizes, fields, seeds, bond dimension, grids) |
| `paper_figures/` | one folder per figure: `prepare_*.py` (results -> `data_raw/`), `make_*.py` and an editable notebook (`data_raw/` -> `output/`), README, `data_raw/source_manifest.json` with SHA-256 hashes; shared style in `_common/` |
| `results/` | empty; every script writes its outputs here |

## Reproducing the paper

The pipeline has four stages.

1. **Simulations** write to `results/`.
   - Exact free fermions at h = 0 (clean and random chains, L up to 512):
     `scripts/run_free_fermion_*.py`, `scripts/run_sdrg_*.py`.
   - DMRG scans in the field, clean chain at h > 0: `scripts/run_clean_ising_gamma_targeted_campaign.py`
     (drives `run_clean_ising_gamma_scan.py`).
   - DMRG scans of the random chain (reentrant regime): `scripts/run_random_ising_gamma_ensemble.py`
     (drives `run_random_ising_gamma_scan.py`), with the parameters of `campaign_specs/`.
   - Exact diagonalization of the periodic rings: `scripts/run_pbc_reentrant_ED_parity.py`.
2. **Analyses** (`scripts/analyze_*.py`, `benchmark_*.py`, `compare_*.py`, `check_*.py`) read
   `results/` and write `summary.json` files next to their outputs.
3. **Figure data**: `paper_figures/<figure>/prepare_*.py` collect the inputs of each figure into
   `data_raw/` and record their hashes in `source_manifest.json`.
4. **Figures**: `paper_figures/<figure>/make_*.py` or the notebook of the same folder draw the figure
   from `data_raw/` only.

The DMRG campaigns are the expensive part (thousands of CPU hours for the random chain). The free-fermion
results take minutes to hours on a workstation.

| paper | folder | data included | main inputs |
|---|---|---|---|
| Fig. 1 | `fig_schematic` | no | one DMRG scan, `results/random_ising_gamma_ensemble/random-h0-Gamma-critical-L16-c64-s0-49-v1` (seed 43) |
| Fig. 2 | `fig_clean_h0` | **yes** | `results/review_v1/free-fermion-h0-v1`, `results/review_v3/clean-fss-systematics-v1` |
| Fig. 3 | `fig_random_h0` | **yes** | `results/review_v1/free-fermion-h0-v1`, `results/review_v1/sdrg-vs-h0-v1` |
| Fig. 4 | `fig_h0_clusters` | no | `compare_sdrg_h0_clusters_review_v1.py` |
| Fig. 5 | `fig_sdrg_purity` | no | `results/review_v1/sdrg-vs-h0-v1`, `results/review_v3/sdrg-disorder-strength-v1` |
| Fig. 6, S3, S4 | `fig_clean_frontier` | no | `results/clean_ising_gamma_campaign` (DMRG) |
| Fig. 7, S1, S2 | `fig_reentrant` | no | `results/random_ising_gamma_ensemble` (DMRG, ED), `results/review_v1/reentrant-domain-walls-v1` |
| Tables, Appendix A | `appendix_numerics` | no | all campaigns |

The README and the docstrings of each `prepare_*.py` name the exact result folders and scripts behind
every panel. To redraw Fig. 2, for example,

    cd paper_figures/fig_clean_h0
    python make_figure.py

## Notes on this copy

The repository was extracted from the working project of the paper with
`scripts/build_github_release_v3.py` (not included). Paths that pointed to working folders of earlier
manuscript versions were remapped to `paper_figures/`, and the rules are listed in `RELEASE_FILES.json`.
Some review checks (`check_*`, `test_*`, `explore_*`, `review_v1_exact_checks.py`) read intermediate
figure data from those folders and have not been re-run in this layout. The production runners, the
analyses and the figure scripts of Figs. 2 and 3 were checked to run from this copy, and Figs. 2 and 3
are reproduced exactly.

## Use of AI tools

Parts of this code were written with the assistance of AI coding tools (OpenAI Codex and Anthropic
Claude, through Claude Code), mainly analysis scripts, figure scripts, data-provenance tooling and
documentation. All code was directed and reviewed by the authors, who take full responsibility for it
and for the results of the paper. The numerical results were validated independently of how the code
was written. DMRG states are compared with exact diagonalization for open and periodic chains with
L = 16 and 17, the free-fermion correlators with an independent LU decomposition, and the figures are
regenerated from raw data whose SHA-256 hashes are recorded in each `source_manifest.json`. Please
report any problem through the issue tracker.

## License

Code under the MIT license (`LICENSE`). The data in `paper_figures/*/data_raw/` under CC BY 4.0
(`LICENSE-DATA`).

## Citation

See `CITATION.cff`. If you use this code, please cite the paper and the archived code,
F. Belchi and N. Statuto, *Zeroth persistent homology of ground-state correlations: code*, Zenodo,
doi:[10.5281/zenodo.23915648](https://doi.org/10.5281/zenodo.23915648).
