# Apéndice A (V1) — tabla de datasets DMRG y protocolo

No es una figura: es la fuente de la Tabla II (`tab:datasets`) y de las macros `App*` del Apéndice A.
`make_numerics_summary.py` lee los `campaign_config.json` y `continuation_summary.npz` de todas las
campañas DMRG usadas en las figuras y escribe `output/numerics_summary.json` (protocolo, bond dimension,
rango y paso de campo, escaneos, puntos y flags por dataset). `tools/make_numbers.py` lo convierte en
`\AppTableRows` y el resto de macros `App*`.

En v1 los datasets a h = 0 aparecen como «check»: validan el protocolo frente a la solución exacta de
fermiones libres, que da todos los resultados a h = 0.

Las comprobaciones de robustez de v1 que también entran en el Apéndice A no se generan aquí, sino con
scripts de `scripts/` cuyos resúmenes lee `make_numbers.py` directamente:

| párrafo del Apéndice A | script | resumen |
|---|---|---|
| Quality flags (χ cerca de la frontera) | `check_chi_convergence_frontier_review_v1.py` | `results/review_v1/chi-convergence-v1/summary.json` |
| Exact solution / Topology (paso doble) | `test_grid_step_review_v1.py` | `results/review_v1/grid-step-v1/summary.json` |
| Reference boundary | `check_digitized_boundary_review_v1.py` | `results/review_v1/digitization-v1/summary.json` |
| Choice of correlator | `test_correlator_choice_review_v1.py`, `test_connected_correlator_review_v1.py` | `results/review_v1/correlator-choice-v1/` |
| Reentrant realizations (umbrales) | `check_reentrant_selection_sensitivity_review_v1.py` | `results/review_v1/reentrant-selection-v1/summary.json` |
