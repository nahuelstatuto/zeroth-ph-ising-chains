# Fig. 5 (review v3): H0 hierarchy against the strong-disorder RG clusters

Half-column figure, two panels stacked vertically. Panel (a) is the former Fig. 3(c) of v2
(same data, same means). Panel (b) is the disorder-strength test C1 of the v3 review
(`documentation/RESULTADOS_C1_DISORDER_STRENGTH_20261029.md`).

OBC, h = 0, Gamma0 = 1, `C = Cstag` raw, `D = 1 - C`, exact free fermions, 300 realizations per point.

- (a) Box ensemble (`fixed_mean`, J in [1, 2]), L = 32-512. Dendrogram purity of the H0 hierarchy, of
  the hierarchy restricted to nearest-neighbour links and of an unrelated partition.
- (b) Power-law family J = u^delta, Gamma = Gamma0 v^delta shifted to fixed mean logarithms, delta =
  0.25-2, L = 32 and 256. Open symbols at "box" repeat the box values of (a), placed at the standard
  deviation of ln J of the box (0.2). delta = 3 is left out because it fails the QC 0 <= C <= 1 for L >= 64.

The figure is drawn at 6.4 x 7.0 in, so that scaled to the column its fonts match the full-width
figures.

```cmd
conda activate tn_tda && python prepare_data.py && python make_figure.py
```

`prepare_data.py` reads `results/review_v1/sdrg-vs-h0-v1/hierarchy_per_realization.npz` and
`results/review_v3/sdrg-disorder-strength-v1/fixed_logmean_d*_L*.npz`; hashes in
`data_raw/source_manifest.json`. Editable notebook: `fig_sdrg_purity.ipynb`. The `sdrg_purity`
block of `output/fig_sdrg_purity_summary.json` feeds the `SdPurity*` macros (tools/make_numbers.py).
