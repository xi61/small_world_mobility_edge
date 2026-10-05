# Experiment catalog and scientific conventions

This catalog records what the checked-in code actually supports.  It is not a
claim that every diagnostic is interchangeable across every graph model.
Defaults in historical modules are preserved because they encode distinct
experiments; use `model_smoke` for a small cross-model integration check.

## Shared diagnostic vocabulary

- **Eigenvalue-density histogram:** bin counts of adjacency eigenvalues,
  normalized to integrate to one.  The ensemble implementation pools retained
  eigenvalues only after each graph's spectral filter is applied.
- **Local spectral-density estimator:** `empirical_spectral_density.py` offers
  a fixed-bandwidth local count, a tail-adaptive variable local count, KDE, and
  a histogram interpolant.  It also supplies a numerical CDF and
  quantile-to-energy conversion.  Graph-level folds are available for
  cross-fitted variable-local-count densities.
- **Local eigenvalue histogram:** a histogram conditional on an energy or
  spectral-quantile window.  Its normalization is conditional on that window;
  it is not the local-count density estimator.
- **Spacing histogram:** a histogram of consecutive gaps from one graph at a
  time.  Raw gaps, per-graph mean-normalized gaps, analytical unfolding, and
  empirical-CDF unfolding are kept distinct in names and manifests.  The code
  never sorts eigenvalues pooled from different graphs to create spacings.
- **Adjacent gap ratio:** `min(delta_i, delta_(i+1)) / max(...)`, computed from
  consecutive within-graph gaps.  It does not require unfolding.
- **Localization:** IPR is `sum_j |v_j|^4`; `N*IPR`, participation ratio,
  participation fraction, maximum coordinate mass, top-coordinate mass, and
  coordinate-decay profiles are reported where eigenvectors are retained.

The quick runner uses full dense eigendecomposition at small `N`; sparse-edge
production modules use `scipy.sparse.linalg.eigsh`.  Small samples that do not
meet a module's support threshold remain marked insufficient rather than being
filled with synthetic statistics.

## Model and pipeline details

### Random simple regular graphs

`regular_baseline.generate_random_regular_graph` calls
`networkx.random_regular_graph`, returning a simple graph with every degree
exactly `k`.  The adjacency matrix is real and symmetric.  One trivial
eigenvalue equal to `k` is removed before bulk spacing analysis.

Implemented diagnostics include the Kesten--McKay density and CDF, analytical
unfolding, midpoint local-density unfolding, local energy-window spacing,
adjacent gap ratios, IPR/participation ratio, and GOE/Poisson comparisons.

```bash
python3 -m small_world_mobility_edge.model_smoke \
  --model regular --n 160 --degree 6 --num-graphs 3 \
  --output-dir outputs/smoke
```

The historical `regular_baseline_experiments` and `local_spacing_by_energy`
modules retain their original settings; the quick runner supplies the small,
parameterized entry point.

### Erdős--Rényi

`erdos_renyi_baseline.generate_erdos_renyi_graph` uses
`networkx.fast_gnp_random_graph` with `p = d/(n-1)`.  Analyses explicitly
select either the full graph or the largest connected component.  Dense bulk
work removes the largest Perron eigenvalue.  The extreme-edge path analyzes
`A/sqrt(d)` and uses dense spectra below a threshold or sparse LA/SA eigenpairs
above it; it separately records central zero multiplicity when computed.

`erdos_renyi_spacing_by_energy` uses separate deterministic seed streams for
density and spacing ensembles, local-count density, empirical CDF unfolding,
fixed-energy and quantile windows, and GOE/Poisson comparisons.  The
extreme-edge module adds bulk/edge windows, localization of retained edge
vectors, theorem-scaling diagnostics, CSV tables, manifests, and plots.

```bash
python3 -m small_world_mobility_edge.model_smoke \
  --model erdos-renyi --n 160 --degree 6 --num-graphs 3 \
  --component-mode largest_connected_component \
  --output-dir outputs/smoke
```

### Chung--Lu

`chung_lu_baseline` constructs either deterministic quantile weights or
sampled Pareto weights.  Its canonical parameter `tau` is the **density**
exponent: the untruncated CCDF exponent is `tau - 1`.  A scaling constant is
solved after clipping so the final expected-weight mean equals the requested
mean degree.  Available upper-cutoff modes are fixed, structural
`c*sqrt(n*d)`, and a power of `n`; diagnostics record clipping and the
admissibility check `max(w)^2 <= sum(w)`.  Graphs are generated with
`networkx.expected_degree_graph(..., selfloops=False)`, so they are simple and
have no self-loops, while realized degrees are random and need not equal the
weights.

The density/spacing pipeline retains its historical full-graph/LCC choice,
seed streams, exact-zero filter, fixed or tail-adaptive local-count estimator,
empirical CDF, quantile windows, and graph-equal aggregation.  Additional
modules provide:

- global density after `lambda/sqrt(d)` versus the semicircle law;
- joint spacing/IPR and coordinate-decay summaries;
- percentile and LCC percentile localization scans;
- eigenvalue-resonance scans near atoms;
- `E=+/-1` projector diagnostics on full graph, LCC, and LCC 2-core;
- sparse degree/spectral-tail matching and a fixed-degree rewired control.

```bash
python3 -m small_world_mobility_edge.model_smoke \
  --model chung-lu --n 160 --degree 8 --tau 2.5 --num-graphs 3 \
  --output-dir outputs/smoke
```

### Fixed realized-degree rewiring control

There is no separate stub-pairing configuration-model implementation in the
local project: no multigraph stub sampler and no “pair then simplify” path was
found.  The implemented control starts from one realized simple Chung--Lu (or
PA) graph and applies a finite `networkx.double_edge_swap` chain.

The control:

- remains an ordinary undirected `networkx.Graph`;
- forbids self-loops and parallel edges;
- preserves every labeled vertex degree and the edge count exactly;
- does not request or guarantee connectedness;
- is not claimed to be a provably uniform sample conditional on the degree
  sequence.

`fixed_degree_configuration_joint_tail` performs paired Chung--Lu versus
rewired comparisons with rank-gap windows, separate mean normalization by
model, density plots, gap ratios, IPR/participation/top-coordinate metrics,
coordinate decay, paired differences, degree-preservation tables, and a run
manifest.

```bash
python3 -m small_world_mobility_edge.model_smoke \
  --model fixed-degree-rewire --n 160 --degree 8 --tau 2.5 \
  --num-graphs 3 --swaps-per-edge 2 --output-dir outputs/smoke
```

### Degree and spectral tail matching

`chung_lu_degree_spectral_tail_match` treats its experimental `--tau` as a
weight-CCDF exponent and records the conversion to the baseline density
exponent (`effective_tau = tau + 1`).  It compares realized degree CCDFs,
absolute non-Perron eigenvalue tails, squared-eigenvalue tails, Hill and
log-regression fits across cutoffs, rank-matched `|lambda|` versus
`sqrt(degree)`, maximum-degree edge scales, sparse residuals, and optionally
the exact-degree rewired control.  Spectral CCDF denominators use `N-1`; the
sparse experiment does not infer an uncomputed central zero multiplicity.

What is matched depends on the experiment:

- Chung--Lu inputs match the requested expected-weight mean and a selected
  input tail convention, subject to finite-size clipping.
- The rewired control matches the **realized labeled degree sequence exactly**.
- PA--Chung--Lu comparisons require compatible `N`, graph count, mean-degree
  target, exponent convention, and sparse selection metadata.  Similar nominal
  exponents alone are not treated as exact degree matching.

```bash
python3 -m small_world_mobility_edge.chung_lu_degree_spectral_tail_match \
  --smoke-test --n 400 --tau 2.5 --degree 8 --num-graphs 3 \
  --k-per-side 40 --target-tail-count 20 \
  --output-dir outputs/smoke/tail
```

### Preferential attachment

The PA subpackage retains BA, affine linear, nonlinear, and Holme--Kim graph
generators plus sparse tail, spacing, localization, rewiring, cached mobility-
edge, extreme-edge, and induced-subgraph diagnostics.  Its [subpackage README](../src/small_world_mobility_edge/preferential_attachment/README.md) gives
the run modes without repeating them here.

The exponent names are intentionally different:

- `alpha`: affine-PA degree-CCDF exponent, with attachment offset
  `delta = m*(alpha-2)`;
- `zeta`: nonlinear attachment-kernel exponent in `degree**zeta`;
- `tau`: Chung--Lu density exponent in the canonical generator;
- selected legacy tail runs use `tau` as a CCDF exponent and record that
  mapping explicitly.

```bash
python3 -m small_world_mobility_edge.preferential_attachment.experiments \
  --smoke-test --output-dir outputs/smoke/pa
```

For the metadata-gated PA--Chung--Lu tail comparison, first create the exact
companion run and then request the comparison:

```bash
python3 -m small_world_mobility_edge.chung_lu_degree_spectral_tail_match \
  --tau 2 --n 400 --degree 8 --num-graphs 3 --k-per-side 40 \
  --target-tail-count 20 --no-run-fixed-degree-control
python3 -m small_world_mobility_edge.preferential_attachment.experiments \
  --smoke-test --run-chung-lu-comparison --output-dir outputs/smoke/pa-matched
```


## Supported model--diagnostic combinations

| Diagnostic | Regular | ER | Chung--Lu | Fixed-degree control | PA | PA disorder / RRG Anderson |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Dense global spectrum/density | yes | yes | yes | yes | selected runs | yes |
| Sparse extreme eigenpairs | no production path | yes | tail path | tail path | yes | selected-window path |
| Local energy/quantile spacing | yes | yes | yes | rank-gap tail | rank-gap and mobility-edge paths | yes |
| GOE/Poisson reference curves | yes | yes | yes | yes | yes | yes |
| IPR/participation versus spectrum | yes | edge path | extensive | paired tail | extensive | extensive |
| Atom/resonance/projector audit | no | zero audit | yes | no | abnormal-band audit | clean-atom policy |
| Degree/spectral-tail fit | no | degree scaling only | yes | paired control | yes | source graph only |
| Top-coordinate/induced-subgraph audit | no | edge vectors | yes | coordinate decay | yes | yes |
| Paired diagonal-disorder sweep | no | benchmark path | no | no | yes | yes |

### Clean affine-PA local modes and diagonal disorder

`preferential_attachment/linear_pa_two_part.py` implements the full-spectrum
clean affine-PA study and its a-posteriori graph-region/local-mode diagnostics.
Its production grid is `N in {500,10000}`, `m=10`, and nominal degree-CCDF
`alpha in {1.05,1.1,1.25,1.5,2,2.5,3}` with one deterministic graph per
condition. Part 2 keeps exact-eigenvector-informed region selection explicit;
capture is not described as an independent prediction.

`random_schrodinger/pa_diagonal_uniform_noise.py` reuses those exact graph
caches and applies `H(a)=A+a*diag(xi)` for raw amplitudes
`{0,0.5,1,2,4,8,16,32}`. One iid uniform field is paired across amplitudes on
each graph. Clean zero-atom solver vectors are masked from primary individual-
state comparisons; positive-disorder states use numerical multiplicity rather
than the clean mask.

`random_schrodinger/anderson_rrg_benchmark.py` is a separate random 3-regular
benchmark with its own graph/potential seed tables and literature convention.
It is not pooled with PA and does not convert the PA graph into a regular graph.

“No production path” means no dedicated historical experiment was found; the
small smoke runner may still calculate a generic diagnostic without implying
historical validation for that model.

## Reproducibility and aggregation

Production modules record explicit graph, weight-sequence, and rewire seed
streams in manifests or graph-level tables.  Gaps and localization statistics
retain graph identifiers through aggregation.  Plotting and CSV/JSON export
live with each experiment module so a checkout can generate new results
without importing from the private source workspace.
