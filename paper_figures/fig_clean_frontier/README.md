# Fig. 5 (V1) y Figs. S3–S4 — frontera de la cadena limpia a h > 0

**Estado en v1 (2026-10-19).** Figuras cerradas.

| figura del PDF | script | notebook | salida |
|---|---|---|---|
| Fig. 5 | `make_figure.py` | `fig_clean_frontier.ipynb` | `output/fig_clean_frontier.{png,pdf}` |
| Fig. S3, primer ciclo H1 | `make_fig_S3_h1_cycle.py` | `fig_clean_frontier_S3.ipynb` | `output/fig_clean_frontier_S3_h1_cycle.*` |
| Fig. S4, robustez frente a τ | `make_fig_S4_h1_tau.py` | `fig_clean_frontier_S4.ipynb` | `output/fig_clean_frontier_S4_h1_tau.*` |

Cambio v1 en la Fig. 5 (referee 2): en (a) los escaneos de rango completo son cuadrados grandes con borde
grueso y las ventanas cuadrados pequeños translúcidos; los inicios H1 son círculos con halo sobre su línea, y
la leyenda separa H0 (L par) de H1 (L impar). Datos: `data_raw/clean_frontier_compact.npz`,
`clean_frontier_oddL_compact.npz` y `clean_frontier_h1_cycle_L31.npz` (`prepare_h1_cycle.py`), con
manifiesto SHA-256. Manuscrito: `figures/fig5_clean_frontier.pdf`, `figS3_…`, `figS4_…`.
Las secciones siguientes documentan la historia de la figura desde v0.

---

# Figura frontera clean h>0 — borrador

Salida: `output/fig_clean_frontier.{png,pdf}` y `output/fig_clean_frontier_summary.json`,
generadas por `make_figure.py`. Dos paneles:

- (a) plano `(h,Gamma)`: máximos de la velocidad del landscape H0 para
  L=16, 32, 64 (cuadrados) y máxima proliferación de ciclos H1 para L=64
  (círculos);
- (b) barrido completo L64 `h=1.7`: respuesta física, `xi_lambda` y `N1`.

## Datos

`data_raw/clean_frontier_compact.npz`, construido por `prepare_data.py` a
partir de 67 `continuation_summary.npz` (SHA-256 de cada fuente en
`data_raw/source_manifest.json`). Por cada `(L,h,chi)`: malla Gamma,
`m^2_{s,B}`, las `L-1` muertes H0 de `D=1-Cstag`, los intervalos H1 (Rips
sobre `clip(1-C,0,2)`, coeficientes Z2, gudhi), desacuerdo de ramas y bond
máximo. Fusión de scans con la regla del análisis previo (Gamma duplicado:
estado de menor energía, nunca promedio). Se usa el chi más alto por `(L,h)`.

Referencia: los 13 puntos digitalizados de la Fig. 1 de Lajkó et al., PRB 101,
024203 (2020); interpolación lineal para `h=0.3,0.7,1.1,1.3,1.5,1.7`.

## Reglas

- H0: derivadas `numpy.gradient(edge_order=2)`, velocidad con todas las capas,
  máximo global refinado con parábola de tres puntos (como en las otras figuras).
- Máscara QC, declarada antes de los cortes full-Gamma de L64: un punto no es
  candidato a máximo si él o un vecino tiene desacuerdo de ramas o saturación
  de bond. Los flags se conservan.
- H1 (definición de `scripts/analyze_clean_H1_threshold_stability.py`):
  `N1(tau)` = número de intervalos H1 con vida `> tau`; estimador = punto medio
  del intervalo de Gamma con mayor `Delta N1/Delta Gamma` positivo, usando solo
  intervalos con ambos extremos válidos en QC. Solo L=64.
- `tau=0.006`: criterio interno del análisis del 2026-10-09, la estimación
  más suave a lo largo de `h` suponiendo una única línea de transición; no se
  eligió por acuerdo con la frontera. Está junto a un cambio de población de
  barras en `h=1.6` (entre 0.006 y 0.007 la estimación pasa de 0.508 a 0.348).
  La alternativa por estabilidad es `tau=0.008` (meseta 0.007–0.010).
- Controles `h>=2(1-1/L)` excluidos. L64 `h=1.95` (chi=512) no convergido: no
  se dibuja (el QC chi=768 murió sin resultados, monitor 2026-10-09).
- Círculos H1 en `Gamma=0`: cortes L64 sin ningún intervalo H1 con vida
  `>tau` en estados válidos (el script lo comprueba; si hubiera ciclos sin
  proliferación, se detiene). En `h=0.3,0.7,1.1` el barrido cubre
  `Gamma=0–1.2`; en los demás solo la ventana simulada.
- Panel (b): se muestra `Gamma>=0.1`; los estados marcados del corte están en
  `Gamma<0.08` y el script comprueba que no hay ninguno en el rango dibujado.

## Números (MAE frente a la frontera, número de cortes; ventanas + full-Gamma)

| L | física | `xi_lambda` H0 | proliferación H1 (`tau=0.006`) |
|---:|---:|---:|---:|
| 16 | 0.166 (9) | 0.055 (9) | — |
| 32 | 0.122 (13) | 0.029 (13) | — |
| 64 | 0.091 (15) | 0.023 (15) | 0.060 (6); 0.032 sin `h=1.3` |

H1 en L64 por corte (`tau=0.006`, frontera entre paréntesis): `h=1.3` 0.455
(0.656), 1.4 0.664 (0.599), 1.6 0.508 (0.453), 1.7 0.385 (0.366), 1.8 0.268
(0.278), 1.9 0.144 (0.154). Con `tau=0.008` cambian solo `h=1.3` (1.125) y
`h=1.6` (0.348). El barrido `h=1.3` terminó después del análisis de
thresholds: es prospectivo y H1 falla allí con cualquier threshold.

## Cautelas para el texto

- Las ventanas dirigidas se diseñaron alrededor de la frontera digitalizada:
  son reconstrucción, no predicción ciega. Los cortes full-Gamma son el test
  prospectivo (H0 L64: MAE 0.017 en cinco cortes).
- En los full-Gamma de L64 hay regiones amplias con desacuerdo de ramas en la
  fase ordenada (probable degeneración Z2); el MAE físico full-Gamma depende de
  la máscara.
- H1 solo existe en la rama de `h` alto (`h>=1.3` en L64). Allí su máxima
  proliferación sigue la frontera para `h>=1.7`, pero no en `h=1.3` ni de
  forma uniforme en `h=1.4–1.6`. Es una señal complementaria, no un
  localizador global. Mecanismo: cuenta inversiones locales de cuatro sitios
  en la jerarquía de correlaciones
  (`documentation/ANALISIS_H1_PERSISTENCIA_TOTAL_CICLOS_CLEAN_L64_20261009.md`).

## Caption (borrador)

> **Clean antiferromagnet in transverse and longitudinal fields (OBC).**
> (a) Phase plane: the line is the thermodynamic boundary digitized from
> Lajkó et al. Squares: maxima of the H₀ landscape velocity ξ_λ for L = 16, 32
> and 64 (slightly offset in h for visibility). Circles: field of maximal
> proliferation of H₁ cycles for L = 64 (joined by the dashed line), i.e. the
> largest increase with Γ of N₁, the number of H₁ intervals with lifetime above
> τ = 0.006. Open circles at Γ = 0: no H₁ interval above τ in any QC-valid
> state of the scan; H₁ appears only for h ≥ 1.3. Thick edges: prospective scans over the full range Γ/J = 0–1.2;
> open symbols: maximum at the edge of the scanned window. (b) Full-range scan
> at L = 64, h = 1.7: physical response −∂_Γ m²_{s,B}, ξ_λ of H₀ and N₁,
> each normalized to its maximum; the dotted line marks the maximal
> proliferation of H₁ cycles and the dashed line the boundary. H₀ maxima are
> refined with a local three-point parabola; states with branch disagreement
> or bond saturation are excluded from the maximum search but kept in the data.

## Versión cerrada (2026-10-16)

La figura vigente es la de `make_figure.py` / `fig_clean_frontier.ipynb`: (a) H0 de L par
(16, 32, 64) e inicio del primer ciclo H1 de L impar (31, 63), con inset cerca del punto
multicrítico (L63, h = 1.9–1.98) y ajuste Γ* = a (h_c − h) para h ≥ 1.9; (b) corte L63 h = 1.7;
(c) correlaciones extremo–extremo y extremo–centro y magnetización del centro. Datos impares en
`data_raw/clean_frontier_oddL_compact.npz` (`prepare_data_oddL.py`, hashes en el manifiesto).
Las secciones anteriores de este README describen la versión con H1 de L64, ya sustituida.

## Figuras suplementarias V1 (2026-10-19, referee 2)

- **Fig. S3, el primer ciclo H1 dibujado** (`make_fig_S3_h1_cycle.py` / `fig_clean_frontier_S3.ipynb`).
  Corte L31, h = 1.7, χ = 128. Datos: `data_raw/clean_frontier_h1_cycle_L31.npz` (`prepare_h1_cycle.py`,
  matrices Cstag del corte completo y flags, hash en el manifiesto). Rips de D = 1 − C hasta dimensión 2
  (GUDHI). Representante: arista de nacimiento del intervalo H1 más largo cerrada por el camino más corto
  (Dijkstra con pesos D) en el grafo D ≤ nacimiento. (a), (b) MDS clásica de D solo para dibujar, en
  Γ = 0.40 (antes del inicio) y 0.50 (después), con las aristas D ≤ punto medio de la vida del ciclo de (b).
  (c) vidas de los dos intervalos H1 más largos frente a Γ, con τ, inicio y frontera.
  Lectura: el subretículo de los extremos forma un cúmulo compacto con los dos extremos; el subretículo
  opuesto (con el centro) forma un arco; el ciclo recorre el arco y cierra por el cúmulo.
- **Fig. S4, robustez frente a τ** (`make_fig_S4_h1_tau.py` / `fig_clean_frontier_S4.ipynb`). Nueve cortes
  h = 1.1–1.9 de L31 y L63 de `clean_frontier_oddL_compact.npz`, misma regla de inicio y máscara QC,
  τ = 0.005–0.15. Error medio y máximo frente a la frontera y cortes sin inicio.

Los notebooks S3/S4 no se sincronizan con `notebook_sync.py to-script` (escribiría `make_figure.py`):
tras editarlos, copiar las celdas al script correspondiente o renombrar el `make_figure.py` generado.
Resúmenes en `output/*_summary.json`, leídos por `tools/make_numbers.py` (macros `Hcyc*`, `Htau*`).
