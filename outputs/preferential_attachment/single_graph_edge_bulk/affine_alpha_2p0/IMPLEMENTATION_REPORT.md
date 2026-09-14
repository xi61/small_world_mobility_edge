# Single-graph edge/bulk implementation report

- Analysis version: `pa_single_graph_edge_bulk_v1`
- Model: `affine_alpha_2p0`
- Canonical parameter: `degree_ccdf_alpha=2`
- Graph: `N=5000`, `m=10`, seed `3479052237`, graph index `0`
- Hoffman job: `14612758`
- Status: `COMPLETE`
- Output directory: `<HOFFMAN_SCRATCH>/small_world_mobility_edge_single_graph_20260901/outputs/affine_alpha_2p0`

## Existing implementations reused

- `preferential_attachment.baseline.generate_preferential_attachment_graph`: canonical affine/nonlinear PA graph.
- `preferential_attachment.full_spectrum_scan`: established dense symmetric adjacency eigensolver constants and histogram centers.
- `preferential_attachment.mobility_edge_postprocess.build_gap_records`: zero-atom removal, atom-boundary segmentation, within-graph gaps, and middle-eigenvalue adjacent-gap ratios.
- Existing held-out `unfolding/crossfit_k5/fold_0.npz`: CDF trained without graph index 0.

## Exact conventions

Eigenvalues are sorted algebraically in ascending order and the same permutation is applied to eigenvector columns.  The saved residual audit verifies `||A v_i-lambda_i v_i||/max(1,|lambda_i|) < 1e-8` for the selected eigenpairs.

The upper-edge windows contain the 10, 25, and 50 largest algebraic eigenvalues and include the Perron eigenpair.  `edge_rank=1` is the Perron eigenvalue, whose two-sided adjacent-gap ratio is `NA`.

Exact zero eigenvalues (absolute tolerance `1e-08`) are excluded from continuous level statistics and split the spectrum into atom-free graph-local segments; no gap or ratio bridges the removed atom.  The bulk-50 window is the feasible 50-level atom-free window, with a neighboring level on both sides, whose center is closest to the median valid spectral rank.  Bulk-10 and bulk-25 are nested around the same reference.

For an interior valid eigenvalue, `r_i=min(lambda_i-lambda_(i-1),lambda_(i+1)-lambda_i)/max(...)`.  Each `r_i` is an individual diagnostic, not a GOE/Poisson distribution test.

The local spacing panels retain the established twelve histogram centers and GOE/Poisson references.  Because one graph supplies only about ten gaps in a width-0.002 cell, the maximum cell width is `0.05` to improve support.  Windows are clipped symmetrically at 0 or 1, so their requested centers remain exact and edge cells are narrower.  Within a window, positive gaps from this one graph are divided by their mean; therefore the normalization denominator is a one-graph, one-window mean.  Unfolding uses the held-out ensemble CDF and does not fit the focal graph.

Top-coordinate support uses the ten largest `|v_i(j)|^2` values.  Vertex IDs are the eigenvector coordinate indices.  Distances are shortest-path distances in the full original graph, while induced-subgraph connectivity is recorded separately.  The random baseline uses `1000` fixed-seed random ten-vertex sets.

## Compact numerical index

- Edge-50 mean NIPR: `73.0507`
- Bulk-50 mean NIPR: `3.83891`
- Edge-50 mean top-10 mass: `0.274376`
- Bulk-50 mean top-10 mass: `0.0330605`
- Edge finite local-r mean (descriptive only): `0.506793`
- Bulk finite local-r mean (descriptive only): `0.445737`
- Generated figures: `28`; see `figure_index.csv`.

No random-Schrodinger/diagonal-disorder perturbation is implemented in this batch.
