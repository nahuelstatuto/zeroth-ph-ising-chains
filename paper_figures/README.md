# Figuras del manuscrito — versión v3 (copiada de v2 el 2026-11-02)

Fuentes de todas las figuras de `paper_manuscript/reviewer/v3/manuscript/`. Copia de `reviewer/v2/manuscript/paper_figures` con dos cambios: Fig. 3 pierde el panel (c), y la nueva Fig. 5 (`fig_sdrg_purity`) lo recoge junto con la pureza frente a δ. Cada carpeta contiene los datos
crudos autocontenidos (`data_raw/` con `source_manifest.json` y hashes SHA-256), los scripts, un notebook
editable que regenera la figura entera, un README y la salida en `output/` (PNG, PDF y `*_summary.json`).
La v0 congelada está en `paper_manuscript/v0/paper_figures/`.

## Figuras del PDF

| PDF | carpeta | notebook | fichero en `figures/` |
|---|---|---|---|
| Fig. 1, esquema | `fig_schematic` | `fig_schematic.ipynb` (fuente final, editado a mano) | `fig1_schematic.pdf` |
| Fig. 2, limpia h = 0 | `fig_clean_h0` | `fig_clean_h0.ipynb` | `fig2_clean_h0.pdf` |
| Fig. 3, aleatoria h = 0 (v3: dos paneles) | `fig_random_h0` | `fig_random_h0.ipynb` | `fig3_random_h0.pdf` |
| Fig. 4, jerarquía H0 y clusters SDRG | `fig_h0_clusters` | `fig_h0_clusters.ipynb` | `fig4_h0_clusters.pdf` |
| Fig. 5, pureza frente a L y δ (v3, media columna) | `fig_sdrg_purity` | `fig_sdrg_purity.ipynb` | `fig5_sdrg_purity.pdf` |
| Fig. 6, frontera limpia h > 0 | `fig_clean_frontier` | `fig_clean_frontier.ipynb` | `fig6_clean_frontier.pdf` |
| Fig. 7, régimen reentrante | `fig_reentrant` | `fig_reentrant.ipynb` | `fig7_reentrant.pdf` |
| Fig. S1, dos fronteras del lóbulo | `fig_reentrant` | `fig_reentrant_S1.ipynb` | `figS1_reentrant_boundaries.pdf` |
| Fig. S2, paredes en la entrada | `fig_reentrant` | `fig_reentrant_S2.ipynb` | `figS2_reentrant_entry_walls.pdf` |
| Fig. S3, primer ciclo H1 | `fig_clean_frontier` | `fig_clean_frontier_S3.ipynb` | `figS3_clean_frontier_h1_cycle.pdf` |
| Fig. S4, robustez H1 frente a τ | `fig_clean_frontier` | `fig_clean_frontier_S4.ipynb` | `figS4_clean_frontier_h1_tau.pdf` |

`appendix_numerics/` no es una figura: genera la Tabla II y las macros `App*` (ver su README).

## Convenciones

- OBC, ensemble `fixed_mean`, operadores de Pauli, `C = Cstag` crudo, topología sobre `D = 1 − C`.
- h = 0: solución exacta de fermiones libres. h > 0: DMRG (TeNPy), sin DMRG nuevo para las figuras.
- Estilo solo desde `_common/paper.mplstyle` y `_common/paper_style.py` (colores por tamaño, física azul y
  círculos, topología naranja y cuadrados, regla de pico de tres puntos).
- Los scripts verifican los hashes de `data_raw/` y nunca lo modifican.

## Flujo de trabajo

1. Editar el notebook de la figura y ejecutarlo (`python _common/notebook_sync.py run <notebook>`).
2. Volcar el notebook al script: `python _common/notebook_sync.py to-script <notebook>` (escribe
   `make_figure.py`; para S3/S4 y S1/S2, que tienen script propio, copiar las celdas al script
   correspondiente).
3. Desde `paper_manuscript/reviewer/v3/manuscript/`: `python tools/sync_figures.py` copia los PDF a
   `figures/` y escribe `figures/MANIFEST.json` con origen y hash; `python tools/make_numbers.py` regenera
   `numbers.tex` desde los `*_summary.json`.

Verificado el 2026-10-19: todas las carpetas tienen README, notebooks ejecutados, `output/` y manifiestos
cuyos hashes coinciden con `data_raw/`.
