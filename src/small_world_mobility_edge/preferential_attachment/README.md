# Preferential-attachment mobility-edge experiment

This subpackage repeats the power-law Chung-Lu tail, spacing, and
eigenvector-localization diagnostics for preferential-attachment graphs.
It leaves all existing Chung-Lu and configuration-model outputs untouched.

## Primary model

The primary generator is NetworkX `barabasi_albert_graph` with:

- `N = 5000`
- `m = 10`
- an explicit complete graph on `m + 1` vertices as the initial graph
- a separate preferential-attachment seed base

With this initial graph the exact edge count is

```text
m * n - m * (m + 1) / 2
```

so the realized mean degree is close to, but not exactly, `2m`. For
`N=5000, m=10`, it is `19.978`.

The standard BA theory comparison uses the degree-CCDF convention

```text
P(D >= k) approximately k^(-tau),  tau = 2.
```

The realized degree-tail exponent is fitted independently for every graph.

## Spectral selections

The production path reuses the validated sparse `eigsh` tail solver. It
computes the largest and smallest algebraic eigenpairs, saves the Perron
eigenvalue separately, and uses full denominator `N-1` for positive-threshold
non-Perron spectral CCDFs.

Spacing and localization use disjoint one-based positive non-Perron edge-gap
rank windows. The default windows are `1-20`, `21-40`, ..., `101-120`.
Every five-graph pilot therefore pools exactly 100 gaps per model and window.
No gap rank appears in two windows.

Sparse production runs do not determine the central zero multiplicity or a
full global density. The required `global_spectral_density.png` is explicitly
labeled as a retained-edge diagnostic in that case. The dense smoke test
records the exact zero multiplicity and validates the global density.

## Rewired control

The paired control calls the existing `rewire_graph_preserving_degrees`
helper. It preserves every labeled degree exactly. Differences are always
defined as rewired minus original. The finite double-edge-swap chain is an
approximate randomized simple graph conditional on the realized labeled
degree sequence; it is not claimed to be a uniform sample.

## Commands

Run from the repository root.

Smoke test (`N=400`, `m=4`, three graphs, full dense validation):

```bash
PYTHONPATH=src python3 -m small_world_mobility_edge.preferential_attachment.experiments --smoke-test
```

Five-graph production-size pilot:

```bash
PYTHONPATH=src python3 -m small_world_mobility_edge.preferential_attachment.experiments --n 5000 --m 10 --num-graphs 5 --run-rewired-control
```

One partial graph job:

```bash
PYTHONPATH=src python3 -m small_world_mobility_edge.preferential_attachment.experiments --n 5000 --m 10 --num-graphs 50 --graph-index 0 --partial-dir outputs/preferential_attachment/barabasi_albert/n_5000_m_10/partials
```

Aggregate all partials:

```bash
PYTHONPATH=src python3 -m small_world_mobility_edge.preferential_attachment.experiments --n 5000 --m 10 --num-graphs 50 --aggregate-partials --partial-dir outputs/preferential_attachment/barabasi_albert/n_5000_m_10/partials
```

Full 50-graph production run (do not launch before reviewing the pilot):

```bash
PYTHONPATH=src python3 -m small_world_mobility_edge.preferential_attachment.experiments --n 5000 --m 10 --num-graphs 50 --run-rewired-control --run-chung-lu-comparison
```

The strict matched Chung-Lu comparison requires an `N=5000`, mean-degree
`20`, degree-CCDF `tau=2.0` run with the same graph count and sparse selection.
When it is missing or incompatible, the BA runner writes a status file and
the exact canonical Chung-Lu generation command instead of comparing
incommensurate outputs.
