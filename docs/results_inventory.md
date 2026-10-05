# Results inventory

This repository is the canonical aggregate for the random-regular,
Erdős--Rényi, Chung--Lu, fixed-realized-degree, and preferential-attachment
work. It contains both runnable scientific code and the compact finalized
results copied back from Hoffman.

## Published result trees

| Model or artifact | Published path | Contents and provenance |
|---|---|---|
| Random simple `k`-regular | `outputs/smoke/regular/` | Verified reference run: graph invariants, empirical density, local spacing/ratios, and localization. No separate large Hoffman regular run was present in the downloaded workspaces. |
| Erdős--Rényi | `outputs/erdos_renyi/` | Density, local-spacing histograms and summaries, largest-component runs, and extreme-edge outputs. |
| Chung--Lu | `outputs/chung_lu/` | Constant-weight and power-law density/spacing outputs, eigenvector scans, exact-`E=1` resonance/projector analyses, joint spacing--IPR results, and degree/spectral-tail matching. |
| Fixed realized-degree control | `outputs/configuration_model/` | Paired Chung--Lu versus exact-labeled-degree double-edge-swap summaries, histograms, and manifest. |
| Preferential attachment | `outputs/preferential_attachment/` | BA pilots; affine-linear and nonlinear production families; rewired controls; coarse/fine mobility-edge summaries; extreme-edge, abnormal-band, and single-graph edge/bulk diagnostics. |
| PA report package | `docs/reports/pa_three_model_edge_bulk_conclusive_report/` | Upload-ready TeX, all 84 referenced recorded figures, tables, a figure manifest, and the compiled PDF. Build auxiliaries and duplicate ZIP/PDF copies are omitted. |
| Clean affine-PA / diagonal-disorder report | `docs/reports/PA_and_Anderson_report_20260924/` | TeX, bibliography, compiled PDF, and 12 recorded figures (11 referenced plus one retained alternate case figure). |

The run manifests and output tables are authoritative for graph counts,
parameters, seeds, component rules, atom filtering, and normalization. Module
defaults may describe a later experiment and should not be used to reinterpret
an older output.

The full clean affine-PA and diagonal-disorder production trees are not copied
into Git because they contain dense eigenvector matrices and per-operator
tables measured in gigabytes. Their complete scientific/report code,
configurations, controlled tests, recorded report figures, and figure-to-input
map are included.

## Research reports

- `docs/eigenvector_mobility_edge.md`: Chung--Lu localization and finite-size
  mobility-edge interpretation.
- `docs/eigenvalue_eigenvector_resonance_alignment.md`: alignment of the
  eigenvalue and eigenvector diagnostics near the `E=1` resonance.
- `docs/preferential_attachment_chung_lu_experiment_handoff.md`: detailed
  PA/Chung--Lu comparison and design history.
- `docs/preferential_attachment_mobility_edge_execution.md`: production
  conventions and original Hoffman execution recipe.
- `docs/preferential_attachment_mobility_edge_execution_report.md`: final PA
  implementation and scientific result report.

These reports have distinct roles and are indexed here rather than repeated in
the README.

## Reproducible but not tracked as Git objects

The local checkout also contains large restart and per-graph products. They are
excluded from the Git publication tier:

- `partials/` and `replay_partials/` task checkpoints;
- per-window `.npz` arrays and cached eigensystem products;
- high-volume per-graph `.csv.gz` tables;
- scheduler logs, Matplotlib/XDG caches, and `.DS_Store` files;
- directories explicitly marked `superseded`;
- the manifest-only interrupted ER `N=5000`, average-degree-3 full-graph run.

These exclusions keep the repository below GitHub's recommended size while
preserving all finalized figures, compact tables, manifests, and reports.
They do not remove a graph generator or analysis path. The Python modules and
the chunk/aggregate utilities under `tools/` reproduce the excluded
intermediates from saved parameters and seeds.

## Provenance and integrity

Hoffman outputs were copied into their existing canonical paths without
renaming result files. Existing files were not overwritten during aggregation.
Where a workflow emitted checksums, software versions, graph IDs, seeds, or SGE
metadata, those records remain with the output. The transfer, audit, validation,
and aggregation entry points are indexed in `docs/hoffman_reproduction.md`.

When publishing a new result, retain its manifest, compact summary tables, and
final figures. Keep its resumable cache beneath the ignored output tree unless
it is intentionally distributed as a separate release asset.
