# Fig. 1 (V1) — de un estado a la velocidad del landscape

**Estado en v1 (2026-10-19).** Figura cerrada. **Fuente final: `fig_schematic.ipynb`**, editado a mano por
el usuario; `make_figure.py` se regenera desde el notebook (`python ../_common/notebook_sync.py to-script
fig_schematic.ipynb`), nunca al revés. Cambio v1 (referee 2): las aristas que fusionan dos componentes son
naranjas entre primeros vecinos y azules (`FAR_COLOR = "#0072B2"`) entre sitios a distancia |i−j| ≥ 2, en la
red y en el barcode. Manuscrito: `figures/fig1_schematic.pdf` (pie en §II). Datos en `data_raw/` con
manifiesto SHA-256; salida en `output/`.

---

# Figura esquema — de un estado a la velocidad del landscape

Salida: `output/fig_schematic.{png,pdf}` y `output/fig_schematic_summary.json`,
generadas por `make_figure.py`. Las maquetas A/B/C de la primera ronda quedan
en `drafts/` como historial de la decisión.

## Mensaje

Una sola realización recorrida por todo el pipeline, en el orden pedido por
el usuario: red del estado → filtración y barcode H0 → landscape →
movimiento del landscape con el campo → velocidad `xi_lambda`. Se toman del
proyecto EmergentEntanglementStructures solo ideas de representación
(instantáneas de la filtración enlazadas al barcode con líneas punteadas,
nodos numerados en círculo con un hueco para la cadena abierta, grosor de
arista según el peso). Datos, estilo y colores son los de este paper.

## Estado elegido

`data_raw/random_h0_L16_seed43_chi64.npz`, copia exacta (SHA-256 en el
manifiesto): random `h=0`, OBC, `fixed_mean`, L=16, seed 43, chi=64,
`Gamma0=0.95`. Criterios objetivos de elección entre las 50 seeds:

- barras H0 heterogéneas en `Gamma0=0.95` (0.153–0.602; en clean L16 todas
  caen en 0.39–0.49 y la filtración resulta poco ilustrativa);
- una fusión que no es entre primeros vecinos (sitios 12–14), así que el
  árbol de fusiones no es simplemente la cadena;
- cero flags de desacuerdo de ramas en todo el barrido.

El H0 se calcula por single linkage sobre `D=1-C` (Kruskal), equivalente al
H0 de Vietoris–Rips. Instantáneas en 12, 7, 3 y 1 componentes
(`eps=0.34, 0.41, 0.53, 0.63`). Pico de `xi_lambda` de esta realización:
`Gamma0*=0.865`.

## Caption (borrador)

> **From a quantum state to the landscape velocity.** One realization of the
> random antiferromagnetic chain (h=0, OBC, L=16, Γ₀=0.95).
> (a) Correlation network of the state: sites are nodes and the edge width is
> proportional to the staggered correlator C_ij = (−1)^{|i−j|}⟨σᶻ_iσᶻ_j⟩.
> ①–④ Filtration of the distance D = 1 − C: an edge enters when
> D_ij ≤ ε, and orange edges are those that merge two connected components.
> (b) H₀ barcode: each merge at ε = d ends one bar [0, d]; the arrow is the
> component that never dies. Dotted lines mark the snapshots ①–④.
> (c) Persistence landscape: each bar becomes a tent of height d/2, and
> λ_k is the k-th largest tent.
> (d) Landscapes at Γ₀ = 0.85 (dashed) and 0.95 (solid): changing the field
> moves the whole landscape; the shaded area is the change of λ₁.
> (e) Landscape velocity ξ_λ of the same realization, the L² norm of
> ∂_{Γ₀}λ normalized by L − 1; the shaded band is the interval shown in (d)
> and the square marks its maximum, the pseudocritical point Γ*₀ of this
> realization.

## Pendiente

- Decidir si el estado ilustrativo es random (actual) o clean.
- Revisar en PDF el tamaño final a doble columna.
