# Chung–Lu

[All models](../../README.md#browse-by-model) · [Full catalog](../experiment_catalog.md#chunglu)

The Chung–Lu experiments introduce heterogeneous expected degrees through
Pareto weights. Baseline `tau` is a density exponent: P(w>x) scales as
x^{−(tau−1)} before clipping. The tail-matching runner instead accepts a CCDF
exponent and explicitly converts it. A common symbol in old plots does not
make these two parameter conventions interchangeable.

The generator rescales clipped weights to the requested expected-weight mean
and uses `networkx.expected_degree_graph(..., selfloops=False)`. Realized
degrees fluctuate. The finite weight cutoff and the check max(w)²≤Σw are
recorded alongside full-graph/LCC selection.

| Experiment | Scientific question | Code in `src/small_world_mobility_edge/` |
|---|---|---|
| Aggregate CDF / local spacing | Does unfolding reveal bulk/edge differences? | `chung_lu_spacing_by_energy.py`, `empirical_spectral_density.py` |
| Joint localization | Do spacing changes coincide with IPR or concentrated coordinates? | `chung_lu_joint_spacing_ipr.py` |
| Resonance / atom diagnostics | Are crowded levels distinct or a repeated eigenspace? | `chung_lu_eigenvalue_resonance_scan.py`, `chung_lu_e1_projector_core_scan.py` |
| Tail matching | How do degree, absolute-eigenvalue, and squared-eigenvalue tails compare? | `chung_lu_degree_spectral_tail_match.py` |

The ±1 projector study compares the full graph, LCC, and LCC 2-core. Projector
diagonals are unchanged by a rotation within a repeated eigenspace; individual
IPRs need not be. This distinction carries into the later PA zero-atom audit.
See the [fixed-degree control](fixed_degree.md) for the paired comparison that
preserves every realized degree rather than only its expected distribution.

```bash
python3 -m small_world_mobility_edge.model_smoke --model chung-lu --n 160 --degree 8 --tau 2.5 --num-graphs 3 --output-dir outputs/smoke
```
