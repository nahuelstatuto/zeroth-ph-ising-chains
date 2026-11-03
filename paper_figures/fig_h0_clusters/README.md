# Figura nueva (V1) — cómo detecta la jerarquía H0 los clusters de la SDRG

Esquema didáctico para §IV-B (revisión v1). Cadena abierta de L = 12 sitios con acoplamientos
elegidos (ver `prepare_data.py`), h = 0, estado fundamental exacto con fermiones libres,
`C = Cstag`, `D = 1 - C`. La SDRG decima el sitio 7 (campo fuerte) y une {4,5,6} y {8,9,10} en
un cluster no contiguo. C(6,8) = 0.97, C(6,7) = C(7,8) = 0.50.

```cmd
conda activate tn_tda && python prepare_data.py && python make_figure.py
```

Hashes en `data_raw/source_manifest.json`. Notebook editable: `fig_h0_clusters.ipynb`.
Los árboles son esquemáticos: la altura de las ramas es el orden de fusión (mismo árbol que con
la escala D; en D el cluster se completa a 0.03 y las mitades por vecinos a 0.50).

**Caption (borrador, inglés):**

> How the $H_0$ hierarchy detects a cluster of the strong-disorder RG, for an illustrative open
> chain of $L=12$ sites at $h=0$ with chosen couplings and its exact ground state. Sites
> 4–6 and 8–10 are strongly coupled, and site 7 carries a strong transverse field. The RG
> decimates site 7 and joins the two halves into one non-contiguous cluster (filled circles).
> (a) Correlation network of the ground state, drawn as in Fig. 1. The cluster is strongly
> correlated across site 7, $C_{68}=0.97$, while $C_{67}=C_{78}=0.50$. (b) $H_0$ hierarchy, the
> single-linkage tree of $D=1-C$. The cluster is one subtree (blue). (c) Hierarchy restricted to
> nearest-neighbour links. The two halves can only merge through site 7, so the cluster never
> appears on its own. Branch heights show the order of the merges.
