# Fixed single-graph edge/bulk PA microscope

This batch uses one stored realization for every output within a model.  The graph is generated once with the canonical PA implementation, the adjacency matrix is diagonalized once, and the identical sorted eigenpairs feed the density, spacing histograms, energy--NIPR plot, upper-edge microscope, matched bulk control, and top-support structural audit.

## Production models

The exact settings are in `production_models.json`:

- affine `degree_ccdf_alpha=1.05`, graph-index-0 seed `4011773341`;
- nonlinear `attachment_zeta=1.50`, graph-index-0 seed `1912200482`;
- affine `degree_ccdf_alpha=2.00`, graph-index-0 seed `3479052237` (control).

Each graph is `N=5000`, `m=10`.  Each uses the existing five-fold CDF file `unfolding/crossfit_k5/fold_0.npz`; its provenance arrays verify that graph index 0 is held out and absent from training.

## Selection and statistics

- Eigenvalues are sorted algebraically ascending, with the identical permutation applied to the eigenvector columns.  Sampled residuals are required to be below `1e-8`.
- Edge top 10/25/50 means the largest algebraic eigenvalues, nested, including Perron. Perron's centered two-gap ratio is `NA`.
- Bulk 50 is the feasible 50-level window inside one zero-atom-free segment whose center is closest to the median valid spectral rank and which leaves a neighboring level on both sides. Bulk 10/25 are nested around that center.
- Exact zero eigenvalues are retained in the saved spectrum/density but excluded from continuous spacing statistics. No gap or ratio crosses the zero atom.
- Each selected eigenvector contributes its ten largest coordinate masses, original vertex IDs, full-graph degrees, induced-subgraph metrics, and full-graph shortest-path distances. A fixed-seed baseline of 1000 random ten-vertex sets supplies z-scores and percentiles.
- The single-graph local histograms keep the established twelve quantile centers, display range, bins, and GOE/Poisson curves. The maximum window width is widened from the ensemble microscope's `0.002` to `0.05` because a single 5000-level spectrum would otherwise contain only about ten spacings per panel; boundary windows are symmetrically clipped so every requested center remains exact. Gaps are constructed only within this graph and divided by the mean positive gap in the same one-graph window.

Random diagonal disorder / a random Schrodinger operator is explicitly outside this batch.

## Run and retrieve

On Hoffman, after syncing the new source and tools, create the fixed log directory and submit once:

```bash
mkdir -p <HOFFMAN_SCRATCH>/small_world_mobility_edge_single_graph_20260901/logs
cd <HOFFMAN_SCRATCH>/small_world_mobility_edge_pa_mobility_20260823/project
qsub tools/hoffman_pa_single_graph_edge_bulk.sh
```

The script is an SGE array with tasks 1--3. Results are written under:

```text
<HOFFMAN_SCRATCH>/small_world_mobility_edge_single_graph_20260901/outputs/
```

From the local project root:

```bash
tools/sync_pa_single_graph_edge_bulk.sh
```

Every model directory includes `run_manifest.json`, `IMPLEMENTATION_REPORT.md`, `figure_index.csv`, compact graph/spectral arrays, one-row-per-selected-eigenpair and one-row-per-support-vertex tables, the random baseline table, and all requested PNG figures.
