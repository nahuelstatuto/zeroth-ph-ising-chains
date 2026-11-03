# Fig. 6 (V1) y Figs. S1–S2 — régimen reentrante

**Estado en v1 (2026-10-19).** Figuras cerradas.

| figura del PDF | script | notebook | salida |
|---|---|---|---|
| Fig. 6 | `make_fig_reentrant_main.py` | `fig_reentrant.ipynb` | `output/fig_reentrant_main.*` |
| Fig. S1, dos fronteras | `make_fig_R1.py` | `fig_reentrant_S1.ipynb` | `output/fig_reentrant_R1_two_boundaries.*` |
| Fig. S2, paredes en la entrada | `make_fig_S2_entry_walls.py` | `fig_reentrant_S2.ipynb` | `output/fig_reentrant_S2_entry_walls.*` |

`make_fig_R2.py`, `make_fig_R3.py` y `make_fig_R4.py` son exploratorias y no entran en el manuscrito.
Manuscrito: `figures/fig6_reentrant.pdf`, `figS1_…`, `figS2_…`. Las secciones siguientes documentan la
historia de la figura desde v0.

---

# Figuras reentrantes — figura principal + R1–R4 (borrador)

OBC, `fixed_mean`, `C=Cstag` cruda, topología sobre `D=1-C`. Cohortes
congeladas: L16 `h=2.5` (97 seeds reentrantes de 1000), L32 `h=2.3125` (137
de 1000). Selección física y cruces congelados. Sin DMRG nuevo: las
imantaciones se midieron sobre las MPS guardadas.

```cmd
conda activate tn_tda && python make_fig_reentrant_main.py && python make_fig_R1.py && python make_fig_R2.py && python make_fig_R3.py && python make_fig_R4.py
```

(ejecutar dentro de `paper_figures_claude\fig_reentrant\`). `prepare_data.py`
reconstruye `data_raw/` desde `results/.../reentrant-exit-anatomy-v1/` y los
arrays congelados; los hashes están en `data_raw/source_manifest.json`.
Contexto y números: `documentation/PROPUESTA_HISTORIA_REENTRANTE_TOPOLOGIA_20261013.md`.

## R1 — dos tipos de frontera (`output/fig_reentrant_R1_two_boundaries`)

**Usada como Fig. S1 del Supplemental Material** (`paper_manuscript/sections/supplement.tex`,
macros `Sup*`). Notebook editable: `fig_reentrant_S1.ipynb`. El resumen incluye
`exit_magnetization_crossing` (cruce |m_s| = m_u frente a la salida y chequeo de
campo medio C(r) ~ m_s^2 - m_u^2 a menos de 0.1 de la salida). Incluye L17/L33
(2026-10-16): imantaciones medidas sobre las MPS guardadas con
`scripts/measure_reentrant_oddL_local_magnetization.py` (validado, |Czz - guardado|
~1e-15, mz idéntico a la caché), añadidas a `data_raw/reentrant_L{17,33}.npz` por
`prepare_data_oddL.py`. Bandas IQR solo para L par. En L impar el cruce |m_s| = m_u
queda algo después de la salida (mediana con signo +0.013 y +0.034); recortar los
extremos no lo corrige de forma sistemática, así que no se le atribuye mecanismo.

> **The two boundaries of the finite-size reentrant lobe are different
> transitions.** OBC random antiferromagnet, L = 16 (h = 2.5, 97 reentrant
> realizations) and L = 32 (h = 2.3125, 137). (a, b) Staggered correlator at
> distance r = L/2 − 1 aligned at each realization's entry and exit (mean;
> band: interquartile range). (c) Shift of the crossing when the correlator is
> measured at another odd distance r (median; band: IQR): the entry does not
> depend on r, the exit does. (d) Jump of −∂E/∂Γ₀ = Σᵢ(Γᵢ/Γ₀)⟨σˣᵢ⟩ across the
> crossing, in units of the realization's typical change per grid step: a kink
> in the energy at the entry, none at the exit. (e) Staggered |m_s| and uniform
> m_u magnetization aligned at the exit: the exit is where the field-induced
> uniform magnetization overtakes the staggered one. (f) Fraction of the
> long-range pairs with positive correlation and median of the strongest pair:
> at the exit half of the pairs are still strongly correlated.

## R2 — qué ve la topología (`output/fig_reentrant_R2_topology`)

> **The H₀ landscape velocity follows the nature of each boundary.** (a, b)
> ln[ξ_λ/median(ξ_λ)] per realization, aligned at its entry and at its exit
> (mean; band: 95 % bootstrap over realizations); in (b) only fields beyond the
> entry are used and grey bands show the same average with exits permuted
> among realizations. (c) Localization error versus the intrinsic width of the
> physical crossing (lobe amplitude over slope), for the H₀ estimate (entry:
> global maximum; exit: second maximum after the post-entry minimum) and for
> the physical crossing itself when r → r ± 2. Lines: binned medians.

## R3 — H1 local (`output/fig_reentrant_R3_H1_local`)

> **H₁ marks where the antiferromagnetic order yields locally.** (a) Coupling
> on the middle bond of the four-site H₁ motifs (lifetime > 0.006, largest
> field) against all bulk bonds. (b) Fraction of reentrant realizations with
> at least one H₁ cycle versus the distance to their exit. (c) Example
> realization (L = 32): motif strength along the chain versus Γ₀, with the
> couplings Jᵢ on the right; motifs appear after the exit on weak bonds.

## R4 — qué dice la topología de la salida (`output/fig_reentrant_R4_exit_topology`)

> **What topology says about the exit.** Averages aligned at each
> realization's exit. (a) The antiferromagnetic skeleton survives: largest
> connected fraction of the graph of odd-distance pairs with positive
> staggered correlation (mean; band: 5–95 %) and mean correlation on the
> edges of the H₀ forest, compared with the half-chain correlator, which
> crosses zero at the exit by definition. The skeleton never covers less than
> half of the chain and is held by short-range antiferromagnetic correlations,
> while the long-range sign changes. (b) Local breaks arrive after the exit:
> fraction of half-chain windows containing a four-site H₁ motif and fraction
> of realizations with at least one H₁ cycle.

Valores en la salida: percolación del esqueleto 0.90 (L16) y 0.98 (L32),
mínimo sobre seeds y campos tras la entrada 0.5; nivel del backbone H0 0.61 y
0.66. La cobertura media por motivos H1 no llega a 0.5 en el rango (máximo
≈0.28 en L16, ≈0.46 en L32).

## Figura principal reentrante (`output/fig_reentrant_main`)

Figura única de dos paneles para el texto principal; R1–R4 pasan a
suplemento. Script `make_fig_reentrant_main.py`; números en
`output/fig_reentrant_main_summary.json`. El borrador anterior de 6 paneles
está en `output/old/`.

**Caption (borrador, inglés):**

> **Topology locates the reentrant entry, not the exit.** OBC random
> antiferromagnet, L = 16 (h = 2.5, 97 reentrant realizations) and L = 32
> (h = 2.3125, 137), individually reentrant out of 1000, and odd chains
> without boundary wall, L = 17 (h = 2.5, 40 of 400) and L = 33
> (h = 2.3125, 43 of 400), drawn to the right of the even size at the same
> field. These are frontier
> points of single realizations, not the disorder-averaged boundary of Lajkó
> et al.: in open chains at these fields the averaged correlator C̄(L/2 − 1)
> stays negative for all Γ0 (non-monotonic, maximum −0.014 at Γ0 ≈ 0.14 for
> L = 16 and −0.015 at Γ0 ≈ 0.2 for L = 32). Long-range (LR) landscape: H0
> barcode of D = 1 − C
> restricted to pairs at distance |d − r| ≤ w, r = ⌊L/2⌋ − 1; level
> M_w = 1 − ⟨death⟩. (a) Frontier points: median over realizations of the
> per-realization estimate, with 95 % (whiskers) and 50 % (thick bars)
> seed-bootstrap intervals; grey bands: 95 % interval of the physical
> boundaries, the zero crossings of the half-chain staggered correlator C(r)
> (for w = 0 the shell is a forest and M_0 ≡ C(r)). The intervals of the odd
> chains are wider because they contain fewer reentrant realizations, about
> 40 instead of 100 to 140. The spread of the individual entries is similar
> for both parities (interquartile range 0.10 to 0.15). Filled symbols, entry;
> open symbols, exit. H0 landscape velocity over all pairs: global maximum
> (entry) and second wave (exit; interior maxima only, 64/97, 20/40, 123/137
> and 37/43 realizations for L = 16, 17, 32 and 33). LR landscape velocity
> (w = 0 all sizes; w = 1 for L = 16, 17, w = 2 for L = 32, 33): maximum
> before (entry) and after (exit) the apex of M_w.
> H1: maximum positive growth of the cycle count N1 (lifetimes > τ = 0.006).
> (b) Normalized landscape velocities of one realization (L = 32, seed 666);
> dotted lines and grey interval: physical entry and exit.

**Números (summary):**

| | L16 | L32 |
|---|---:|---:|
| entrada física (mediana, IC95) | 0.118 [0.104, 0.131] | 0.138 [0.112, 0.161] |
| velocidad H0 global | 0.081 [0.069, 0.106] | 0.087 [0.087, 0.112] |
| LR w=0 | 0.106 [0.094, 0.131] | 0.137 [0.112, 0.162] (127/137) |
| LR w=1 (L16) / w=2 (L32) | 0.106 [0.087, 0.119] (95/97) | 0.137 [0.112, 0.162] (124/137) |
| MAE entrada seed a seed: H0 / w0 / w1-w2 / constante | 0.027 / 0.014 / 0.016 / 0.056 | 0.068 / 0.014 / 0.014 / 0.080 |
| salida física | 0.420 [0.394, 0.443] | 0.479 [0.462, 0.493] |
| segunda ola H0 (64/97, 123/137) | 0.519 [0.469, 0.544] | 0.487 [0.462, 0.512] |
| LR w=0, máximo tras el ápice | 0.331 [0.294, 0.356] | 0.387 [0.362, 0.412] |
| LR w=1 (L16) / w=2 (L32), máximo tras el ápice | 0.319 [0.294, 0.344] | 0.337 [0.337, 0.387] |
| H1, máx. crecimiento de N1 (95/97, 137/137) | 0.431 [0.406, 0.481] | 0.438 [0.438, 0.462] |
| MAE salida seed a seed: H0 / LR w0 / LR w1-w2 / H1 / constante | 0.097 / 0.102 / 0.118 / 0.111 / 0.072 | 0.096 / 0.079 / 0.109 / 0.095 / 0.071 |

Cautelas para el texto:

- Entrada: el landscape LR coincide con la física en L32 y queda a un paso de
  malla en L16 (mediana 0.106 frente a 0.118). No es un sesgo real: 54/97
  seeds caen en el mismo intervalo, 24/97 un paso antes, sesgo mediano −0.003;
  la velocidad solo toma puntos medios de la malla y el cruce físico se
  interpola.
- Salida: todas las estimaciones topológicas fallan de forma distinta según
  el tamaño. LR (w=0 y anchos) llega antes que la física; H0 tarde en L16 y
  dentro de la banda en L32; H1 dentro en L16 y antes en L32. Ninguna bate a
  la constante seed a seed. Mensaje: la topología ve reorganizaciones
  alrededor de la salida, pero no fija su posición.
- La identidad M_0 ≡ C(r) se menciona en el caption solo para explicar por
  qué la física y el landscape w=0 hablan del mismo shell; no se usa como
  detector.
- El panel (b) es un ejemplo en el que la velocidad w=0 tiene un pico en
  ambas fronteras (salida a un paso en el 36 % de las seeds de L32 y en el 8 %
  de L16, siempre en salidas nítidas). No es el caso típico.

**Justificación física (para el texto).** A larga distancia
C_ij ≈ (−1)^{i−j} m_i m_j; con m_i = m_u + (−1)^i m_s, la media sobre pares
da C(d impar) ≈ m_s² − m_u² y C(d par) ≈ m_s² + m_u² > 0.
La salida es el relevo suave (anchura ~0.2 frente a ~0.013 en la entrada) de
la imantación staggered por la uniforme, y solo cambia de signo la media de los
pares a distancia impar (en la salida 53–54 % de los pares siguen positivos,
el más fuerte ≈0.55–0.61).
1. **H0 global no la ve** porque su bosque usa las correlaciones más fuertes
   (d = 1, 2), que siguen grandes: el backbone vale 0.61–0.66 en la salida y la
   red AFM de signo conecta el 90–98 % de la cadena. La salida no fragmenta el
   orden de corto alcance; H0, que resume extremos, no puede marcar el cero de
   una media.
2. **Los shells w ≥ 1 no la ven por su nivel** porque incluyen distancias pares, positivas
   por la imantación uniforme; su nivel queda en 0.3–0.4 en la salida
   (detalle en `output/old/fig_reentrant_main`, paneles d–e).
3. **H1 no la ve** porque las roturas locales (motivos de cuatro sitios sobre
   los enlaces J más débiles) llegan después: cubren el 8 % (L16) y el 22 %
   (L32) de las ventanas de media cadena en la salida y crecen tras ella
   (R4).
4. La entrada sí es visible porque es abrupta (reorientación de dominios de
   Néel, salto de dE/dΓ0) y reordena el barcode entero.

Lo que la topología aporta: localiza la entrada seed a seed; identifica la
salida como cruce de media de una distribución heterogénea con un esqueleto
AFM superviviente; muestra que la frontera vive solo en la escala de media
cadena y en distancias impares; y cartografía dónde se rompe luego el orden
(H1 sobre enlaces débiles).

## Números clave

| | L16 | L32 |
|---|---:|---:|
| anchura del cruce entrada / salida | 0.012 / 0.217 | 0.015 / 0.190 |
| salto de `-dE/dGamma0` entrada / salida | 3.1 / 0.87 | 2.0 / 0.90 |
| H0 alineado en salida (log) vs nulo 95 % | 0.17 vs ≤−0.01 | 0.22 vs ≤0.05 |
| J de los motivos H1 / todos (p) | 1.26 / 1.50 (3e-13) | 1.24 / 1.50 (5e-37) |
| seeds con H1 en salida−0.10 / 0 / +0.05 | 26 / 61 / 77 % | 54 / 82 / 93 % |

## Pendiente

- La parte reentrante no está cerrada: no promover a `paper_figures/`.
- Controles ED con la unión v2+v3 cuando la v3 vuelva a estar disponible.
- Tendencia de la salida con L: ver
  `documentation/PLAN_REENTRANT_EXIT_TREND_20261014.md`.

## L impar (2026-10-16)

La figura principal (`make_fig_reentrant_main.py`, notebook `fig_reentrant.ipynb`) incluye
ahora, a la derecha de cada tamaño par, la cohorte impar sin pared de borde: L17 h = 2.5
(40 reentrantes de 400) y L33 h = 2.3125 (43 de 400). Datos en `data_raw/reentrant_L{17,33}.npz`,
construidos por `prepare_data_oddL.py` desde la selección congelada aplicada en
`scripts/analyze_reentrant_oddL_L17_L33.py`, con las mismas definiciones de H0, landscape LR y H1.
Resultados: `documentation/RESULTADOS_REENTRANT_ODDL_20261016.md`. R1–R4 siguen solo con L par.
El texto de la §VI no se ha reescrito (decisión del usuario).

---

## v3 (2026-11-03): panel (c), paridad de paredes del lóbulo promediado (R1-4)

`prepare_parity_average.py` genera `data_raw/reentrant_parity_average.npz`, con las curvas C(r=7) por realización de todas las realizaciones a h = 2.5: anillos ED L16/L17 (seeds 5000-5999, pbc-reentrant-ED-parity-v1) y cadenas abiertas DMRG L16 (1000) y L17 (400) de producción. El hash está en source_manifest.json. `make_fig_reentrant_main.py` dibuja la figura en una columna: (a) arriba y (b), la paridad, abajo (la antigua (b), una realización, se ha eliminado). En (b) el azul son paredes en pares y el naranja una pared forzada; la línea continua es anillo y la discontinua, abierta. Documentación: `documentation/D3_RECORTE_SEC_VI_20261103.md`.
