# Preferential attachment

[All models](../../README.md#browse-by-model) · [Disorder extensions](disorder_and_anderson.md)

## Models and parameter conventions

The generator supports BA, affine linear PA, nonlinear PA, and Holme–Kim.
The later geometry, hub, and multifractality work uses **affine linear PA**:
start with K_{m+1}; each new vertex attaches to m distinct existing vertices
by weighted sampling without replacement, with weights kᵢ+δ computed before
that growth step. The graph is simple and undirected. For m=10,

$$|\mathcal E|=mN-\frac{m(m+1)}2=10N-55,\qquad
\bar d=20-\frac{110}{N}.$$

The degree parameter α denotes the nominal **CCDF** exponent, not the degree
probability-mass exponent. The continuum calculation is

$$\frac{d(k+\delta)}{dt}\simeq
\frac{m(k+\delta)}{(2m+\delta)t},\qquad
k(t)+\delta\simeq(m+\delta)(t/s)^{m/(2m+\delta)}.$$

Uniform arrival times s then give a tail proportional to
(k+δ)^{−(2+δ/m)}, hence α=2+δ/m and δ=m(α−2). The mass-function exponent is
α+1. This is an asymptotic approximation; finite-N sampling without replacement
and the seed clique affect measured tails. At α=1.05, δ=−9.5 and all eligible
vertices still have positive attractiveness because their degree is at least 10.
BA is the δ=0, α=2 case. Nonlinear PA uses a separate kernel exponent ζ; ζ is
not interchangeable with α.

## Experiments

| Study | Settings | Page / code |
|---|---|---|
| Model-family / ensemble scans | BA, affine, nonlinear, Holme–Kim; grids recorded in individual runs | `preferential_attachment/experiments.py`, [catalog](../experiment_catalog.md) |
| Three-model single-graph edge/bulk | N=5000, m=10; affine α=1.05, affine α=2, nonlinear ζ=1.5 | [Geometry](../experiments/single_graph_geometry.md), [configuration](../../experiments/single_graph_edge_bulk/production_models.json) |
| Clean affine two-part study | N=500 and 10000; α=1.05,1.1,1.25,1.5,2,2.5,3; one graph per condition | [Hub interactions](../experiments/hub_interactions.md), [configuration](../../experiments/linear_pa_two_part_20260924/config.json) |
| October exact-level audit | Original α=1.05 N=5000 and N=10000 graphs | [Geometry and near-zero structure](../experiments/single_graph_geometry.md) |
| Generalized-moment scaling | Reuse the 14 clean two-part eigensystems | [Multifractality](../experiments/pa_multifractality.md) |
| Diagonal disorder | Fixed clean graphs; paired uniform potential amplitudes | [112-operator sweep](../experiments/pa_diagonal_disorder.md) |

The N=5000 audit graph is not a third size in the two-size multifractal fit.
It belongs to the earlier three-model experiment. Vertex IDs are meaningful
only within their saved realization; “vertex 40” is not a matched hub across
different graphs.
