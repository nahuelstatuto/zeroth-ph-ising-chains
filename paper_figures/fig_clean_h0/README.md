# Fig. 2 (V1) — cadena limpia a h = 0 con fermiones libres exactos

Sustituye a la Fig. 2 de la v0 (DMRG, L = 8–128), archivada en `old_v0/`; esos datos DMRG se
usan como validación en el Apéndice A (`scripts/review_v1_exact_checks.py`). Motivo: a h = 0 la
cadena es exactamente soluble (fermiones libres), y todo h = 0 se presenta con la solución exacta;
el DMRG se reserva para h > 0.

OBC, J = 1, `C = Cstag` cruda, topología sobre `D = 1 - C`, L = 16–512, malla g ∈ [0.5, 1.5]
con paso 0.005 (`src/clean_tfim_tn/exact/free_fermion_h0.py`). ξ_λ con la forma cerrada de la
Ec. (6). Ajustes en el panel (c): g* = g_c + a/L + b/L² sin pesos, L ≥ 32; en el resumen,
también a/L con L ≥ 64 y exponente libre.

```cmd
conda activate tn_tda && python prepare_data.py && python make_figure.py
```

Fuentes: `results/review_v1/free-fermion-h0-v1/clean_L*.npz` y `clean_L512_full.npz`
(`scripts/run_free_fermion_clean_L512_full_review_v1.py`); hashes en `data_raw/source_manifest.json`.
Notebook editable: `fig_clean_h0.ipynb`.

**Caption (borrador, inglés):**

> Clean chain at $h=0$ with open boundaries, exact free-fermion ground states for $L=16$ to
> $512$. (a) Field response $-\partial_g m^2_{s,\mathcal B}$ of the squared staggered
> magnetization of the central half of the chain. (b) Landscape velocity $\xi_\lambda(g,L)$.
> Symbols mark the maxima, located with the rule of Sec. II, and the dashed line is the exact
> critical point $g_c=1$. (c) Pseudocritical points against $1/L$, with unweighted fits
> $g^\star=g_c+a/L+b/L^2$ to $L\geq32$ (open symbols, $L=16$, not fitted).

**Review v3 (B5).** Panel (c) and all fits now use the maxima refined off the 0.005 grid
(`data_raw/clean_h0_refined_peaks.npz`, from `results/review_v3/clean-fss-systematics-v1/summary.json`,
script `scripts/analyze_clean_fss_systematics_review_v3.py`). The grid biases the maxima of xi_lambda by
up to 6e-4 at L = 512. The summary stores the systematics over ansatzes and fit windows (`fits`) and the
fits of the grid maxima (`fits_grid`). Panels (a) and (b) still mark the grid maxima.
