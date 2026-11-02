# Fixed-degree controls and tail matching

[All models](../../README.md#browse-by-model) · [Full catalog](../experiment_catalog.md#fixed-realized-degree-rewiring-control)

The experiments called “configuration” in parts of the research history use
a **simple, degree-preserving double-edge-swap chain**, not a stub-paired
multigraph. The source graph and its rewired counterpart have identical
labeled degrees and edge counts. The finite chain does not guarantee uniform
sampling of all simple graphs with that sequence or preserve connectedness.

| Comparison | What is held fixed | Implementation |
|---|---|---|
| Chung–Lu versus rewired | Realized degree at every vertex | `fixed_degree_configuration_baseline.py`, `fixed_degree_configuration_joint_tail.py` |
| PA versus rewired | PA realization's labeled degree sequence | `preferential_attachment/mobility_edge_rewiring_comparison.py` |
| Degree versus spectral tails | Same realization and explicit spectral selection | `chung_lu_degree_spectral_tail_match.py`, PA tail diagnostics |

Code paths are relative to `src/small_world_mobility_edge/`. Matching a nominal
power-law exponent across two model families is weaker than matching the
actual degree sequence. Mean degree, cutoff, N, graph count, and eigenvalue
selection also change the comparison.

A star with hub degree d has eigenvalues ±√d. This motivates comparing |E|
with √d and E² with d. If that approximation describes a tail and the degree
CCDF scales as k^{−α}, the corresponding |E| CCDF scales as x^{−2α} and the
E² CCDF as y^{−α}. Interacting hubs and finite cutoffs can invalidate a direct
rank-by-rank correspondence. The implemented diagnostics report several fit
cutoffs, residuals, and rank matches rather than treating this heuristic as an
identity for every eigenpair.

```bash
python3 -m small_world_mobility_edge.model_smoke --model fixed-degree-rewire --n 160 --degree 8 --tau 2.5 --num-graphs 3 --swaps-per-edge 2 --output-dir outputs/smoke
```
