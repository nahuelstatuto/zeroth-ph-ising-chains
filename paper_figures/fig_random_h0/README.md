# Fig. 3 (V1, two panels since review v3) — cadena aleatoria a h = 0 con fermiones libres exactos

Sustituye a la Fig. 3 de la v0 (archivada en `old_v0/`; la v0 original está en
`paper_manuscript/v0/paper_figures/fig_random_h0/`). Motivo: revisión v1 (ver
`documentation/REVISION_REFEREES_V1_20261016.md`, secciones 4 y 6b).

OBC, `fixed_mean`, `C = Cstag` cruda, topología sobre `D = 1 - C`, fermiones libres exactos
(`src/clean_tfim_tn/exact/free_fermion_h0.py`, validado contra ED y DMRG). 1000 realizaciones
para L = 16–128 y 500 para L = 256 (paso 0.005); comparación H0–SDRG con 300 realizaciones para
L = 32–512 en Γ0 = 1.

```cmd
conda activate tn_tda && python prepare_data.py && python make_figure.py
```

`prepare_data.py` lee `results/review_v1/free-fermion-h0-v1/` y
`results/review_v1/sdrg-vs-h0-v1/hierarchy_per_realization.npz`; hashes en
`data_raw/source_manifest.json`. Notebook editable: `fig_random_h0.ipynb`.

**Caption (borrador, inglés):**

> Random chain at $h=0$ with open boundaries and fixed-mean disorder, exact free-fermion
> ground states. (a) Distribution of the topological pseudocritical points
> $\Gamma^\star_{0,\alpha}$, the maxima of $\xi_\lambda$ of individual realizations, for
> $L=16$ to $256$ (1000 realizations, 500 for $L=256$). Vertical lines mark the means and the
> dashed line the critical point $\Gamma_{0,c}=1$. (b) Width of the distributions of the
> topological (squares) and physical (circles) pseudocritical points against $1/L$, with 95%
> bootstrap intervals. The dotted line is the infinite-disorder scaling $L^{-1/2}$.
> (c) Agreement between the $H_0$ hierarchy of each realization and its strong-disorder RG
> clusters at $\Gamma_0=1$, measured by the dendrogram purity, for the $H_0$ hierarchy
> (squares), the hierarchy restricted to nearest-neighbour links (triangles) and an unrelated
> partition (dotted), 300 realizations per size, 95% bootstrap intervals.

**Review v3.** Panel (c) moved to `../fig_sdrg_purity` (panel (a) there). The figure keeps (a) and (b),
as a half-column figure, panels stacked, at 6.4 x 7.0 in. `data_raw/random_h0_sdrg_hierarchy.npz` is unchanged and still hashed, but no longer
plotted. The caption above keeps (c) only as history; the current caption is in `sections/04_random_h0.tex`.
