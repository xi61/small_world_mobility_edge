# Exact-spectrum and graph-anatomy reproduction

These are the October follow-up scripts, parameterized for a public checkout.
Their named modes and indices intentionally refer to the recorded α=1.05
N=5000 and N=10000 realizations. They are **not generic scripts for arbitrary
graphs with the same N**. Supplying a different realization would invalidate
the fixed annotations. The full audit verifies graph and spectral identities;
run it before interpreting those annotations.

## Input layout

Let `ARCHIVE_ROOT` contain the original run-relative tree:

```text
ARCHIVE_ROOT/
  outputs/
    preferential_attachment/single_graph_edge_bulk/affine_alpha_1p05/
      spectrum/single_graph_eigenpairs.csv.gz
      spectrum/single_graph_spectral_arrays.npz
      tables/{graph_realization.npz,selected_eigenvectors.npz,...}
      run_manifest.json
    linear_pa_two_part_20260924_v3/cases/N_10000/alpha_1p05/
      arrays/{graph_edges.npz,full_eigenvalues_ascending.npy,full_eigenvectors_ascending.npy,...}
      tables/full_spectrum_localization.csv.gz
      part2/...
      graph_cache_manifest.json
      spectral_cache_manifest.json
    pa_diagonal_uniform_noise_20260924_v1/cases/N_10000/alpha_1p05/report/
      tables/...
      run_manifest.json
  audit_outputs/pa_bulk_peak_readiness_20261004/...
  src/small_world_mobility_edge/preferential_attachment/single_graph_edge_bulk.py
```

The historical cache audit additionally requires saved residual arrays,
graph metadata, part-2 manifests/tables, energy-bin edges/density/localization,
and the four readiness files listed in its `paths` dictionary. It hashes these
inputs as provenance. Its error lists any missing input before computation.
The readiness files are historical evidence, not executable dependencies of
the numerical formulas; they remain required for this exact audit replay.
Full caches and this historical archive are not bundled in Git.

## Cache-only full audit

```bash
python3 tools/audit_pa_noncentral_targets.py \
  --root "$ARCHIVE_ROOT" --output outputs/noncentral_audit
```

It reproduces histogram controls, exact level/rank/gap tables, shell masses,
overlaps, star-template tests, and inside/outside residuals without generating
a graph or diagonalizing an operator. It refuses an existing output directory
unless `--overwrite` is supplied. Its [recorded compact results](../../docs/results/pa_spectral_anatomy/)
can be read without the archive.

## N=10000 energy, shell, and hub figures

```bash
python3 tools/pa_spectral_anatomy/make_pa_followup_figures.py \
  --case-dir "$ARCHIVE_ROOT/outputs/linear_pa_two_part_20260924_v3/cases/N_10000/alpha_1p05" \
  --audit-dir docs/results/pa_spectral_anatomy \
  --output-dir outputs/anatomy/n10000
```

This reads the saved graph/eigenvector arrays and audit tables. It writes
PNG/PDF figures, exact-level CSV, and standalone HTML. HTML templates are
packaged beside the script, not expected inside the output directory.

## Table-only figures

```bash
python3 tools/pa_spectral_anatomy/make_n5000_spectrum_visual.py \
  --source-table "$ARCHIVE_ROOT/outputs/preferential_attachment/single_graph_edge_bulk/affine_alpha_1p05/spectrum/single_graph_eigenpairs.csv.gz" \
  --output-dir outputs/anatomy/n5000
python3 tools/pa_spectral_anatomy/make_n5000_zero_retained_histogram.py \
  --source-table "$ARCHIVE_ROOT/outputs/preferential_attachment/single_graph_edge_bulk/affine_alpha_1p05/spectrum/single_graph_eigenpairs.csv.gz" \
  --output-dir outputs/anatomy/n5000
python3 tools/pa_spectral_anatomy/make_zero_retained_histogram.py \
  --source-table "$ARCHIVE_ROOT/outputs/linear_pa_two_part_20260924_v3/cases/N_10000/alpha_1p05/tables/full_spectrum_localization.csv.gz" \
  --output-dir outputs/anatomy/n10000
```

These preserve the original source-table schemas. The public ranked-level
CSVs have renamed fields for readability and are not drop-in replacements.

## Near-zero signed-connector analysis

```bash
python3 tools/pa_spectral_anatomy/analyze_zero_crossing_modes.py \
  --archive-root "$ARCHIVE_ROOT" --output-dir outputs/anatomy/nearzero
```

This analyzes nonzero levels in each graph's zero-crossing histogram bin,
selected using its original filtering and Freedman–Diaconis edges. Some
N=5000 eigenvectors were not saved. The public default stops if they are
missing. Add `--recover-missing-vectors` only when deliberately permitting
targeted shift-invert eigensolves on the saved adjacency. Recovered energy,
IPR, and residual must match the retained scalar records; no graph is
regenerated. This option can be computationally expensive and was not used
for the publication-update checks.
