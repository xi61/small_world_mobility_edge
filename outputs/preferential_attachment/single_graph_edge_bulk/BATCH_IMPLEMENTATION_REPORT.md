# Fixed single-graph edge/bulk implementation report

## Completion

All three `N=5000`, `m=10` production realizations completed on Hoffman job array `14612758`. The transferred local output root is `outputs/preferential_attachment/single_graph_edge_bulk`; the authoritative Hoffman root is `<HOFFMAN_SCRATCH>/small_world_mobility_edge_single_graph_20260901/outputs`. No production task failed or remained incomplete. The combined figure index contains 84 PNG files.

## Files added and existing code reused

Added: `single_graph_edge_bulk.py`, the Hoffman CLI/array/sync tools, `experiments/single_graph_edge_bulk/` inventory, focused tests, and this summarizer. No established source file was rewritten or modified.

The batch reuses the canonical `generate_preferential_attachment_graph`, the established dense symmetric adjacency eigensolver convention, `build_gap_records` atom/gap bookkeeping, the historical histogram quantile centers/bins and GOE/Poisson references, and each model's existing graph-fold-0 held-out CDF.

## Exact conventions

Eigenvalues are sorted algebraically ascending and the same permutation is applied to eigenvector columns. Sampled selected/quartile eigenpair residuals must be below `1e-8`. Edge top 10/25/50 are nested largest-algebraic windows and include Perron; Perron's two-sided ratio is `NA`. Bulk 50 is a 50-level window inside one atom-free segment, centered as close as feasible to the median valid rank while retaining one external level on either side; bulk 10/25 are nested at the same center.

Exact zero eigenvalues are retained in the saved spectrum/density but removed from continuous spacing statistics and split the spectrum; no gap or ratio crosses the atom. For each interior valid eigenvalue, `r_i=min(lambda_i-lambda_(i-1),lambda_(i+1)-lambda_i)/max(...)`. Every `r_i` is only an individual local-spacing diagnostic.

The local one-graph histograms preserve the established twelve centers, display bins, unfolding, and references. To obtain more support than a width-0.002 single-graph cell, the maximum width is 0.05; boundary cells are symmetrically clipped so their requested center remains exact. Every gap remains within one graph and is normalized by the mean positive gap of that same one-graph window.

Top support means the ten largest `|v_i(j)|^2` coordinates, where coordinate `j` is original vertex `j`. Induced-subgraph connectivity and full-graph shortest-path distance are stored separately. Comparisons use 1000 fixed-seed random ten-vertex sets. Random diagonal disorder was not implemented.

## Parameters and compact descriptive results

| model | seed | zero atom | edge-50 mean NIPR | bulk-50 mean NIPR | edge-50 mean top-10 mass | bulk-50 mean top-10 mass | edge-50 mean local r | bulk-50 mean local r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| affine_alpha_1p05 | 4011773341 | 1844 | 496.54 | 49.8905 | 0.467017 | 0.231001 | 0.532419 | 0.529287 |
| nonlinear_zeta_1p5 | 1912200482 | 240 | 313.371 | 10.7532 | 0.438568 | 0.0797995 | 0.53691 | 0.587892 |
| affine_alpha_2p0 | 3479052237 | 0 | 73.0507 | 3.83891 | 0.274376 | 0.0330605 | 0.506793 | 0.445737 |

### Edge-50 structural support versus bulk-50

Negative distance z-scores mean closer support than random ten-vertex sets; positive density/degree z-scores mean denser or higher-degree support.

| model | edge/bulk mean support degree | edge/bulk induced density | edge/bulk mean distance | edge distance z | edge density z | edge mean-degree z |
|---|---:|---:|---:|---:|---:|---:|
| affine_alpha_1p05 | 802.134/10.568 | 0.433778/0.0257778 | 1.568/2.05378 | -10.6632 | 28.7117 | 22.8376 |
| nonlinear_zeta_1p5 | 696.21/10.076 | 0.389333/0 | 1.61467/2.00356 | -10.7194 | 23.8638 | 17.8791 |
| affine_alpha_2p0 | 178.426/10.766 | 0.351556/0 | 1.65111/2.93156 | -11.8796 | 36.833 | 23.6671 |

These values are descriptive comparisons of fixed windows from one realization per model. They do not establish a GOE or Poisson distribution for a top-10/25/50 set.

## Output index

- `batch_summary.csv`: complete numerical batch summary, including all nested window sizes.
- `batch_figure_index.csv`: every generated figure and exact path.
- Each model: `run_manifest.json`, `IMPLEMENTATION_REPORT.md`, `figure_index.csv`, graph/spectral arrays, selected eigenvectors, edge/bulk eigenpair table, normalized support table, random baseline, residual audit, representative comparison table, and plots.
- Transfer logs were used for validation but are not included in the Git publication tier.
