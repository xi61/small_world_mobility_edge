# Preferential-attachment mobility-edge production

This workflow keeps legacy outputs unchanged and writes all new results below
`outputs/preferential_attachment/mobility_edge/`.

## Scientific conventions

- Affine PA uses the canonical `degree_ccdf_alpha`, with
  `P(D >= k) ~ k^(-alpha)` and `delta = m(alpha - 2)`.
- Nonlinear PA uses `attachment_zeta`, with attachment weight `degree^zeta`.
- Every raw gap is formed within one graph and assigned at its eigenvalue
  midpoint. Every raw adjacent-gap ratio is assigned to its middle eigenvalue.
- Raw adjacent-gap ratio is primary. Five-fold, graph-level cross-fitted
  empirical-CDF spacing histograms are secondary.
- Exact zero atoms are removed from the continuous scan and reported
  separately. Inclusive and positive-gap-conditioned raw ratios are both
  retained.
- A displayed spacing histogram must have at least 100 positive gaps. An
  adaptively widened histogram records its actual bounds.

## Protected legacy cache

The original read-only cache is:

```text
<HOFFMAN_SCRATCH>/small_world_mobility_edge_pa_full_20260813
```

Its verified durable copy is:

```text
<HOFFMAN_HOME>/small_world_mobility_edge_cache/pa_full_20260813
```

Audit manifests and SHA-256 indexes are below:

```text
<HOFFMAN_HOME>/small_world_mobility_edge_cache/pa_full_20260813_provenance
```

Run `tools/audit_pa_legacy_cache.py` again after any transfer. Never write
new analysis products into the legacy cache tree.

## Hoffman environment

Production uses Grid Engine/SGE. The validated Hoffman environment is module
`python/3.9.6` with NumPy 1.22.2, SciPy 1.7.1, NetworkX 2.6.2, and
Matplotlib 3.4.3:

```bash
export PA_PROJECT_ROOT=<HOFFMAN_SCRATCH>/small_world_mobility_edge_pa_mobility_20260823/project
export PA_OUTPUT_ROOT=<HOFFMAN_SCRATCH>/small_world_mobility_edge_pa_mobility_20260823/outputs/preferential_attachment/mobility_edge
export PA_PYTHON=python3
```

Each production array has 100 tasks and five graphs per task. A task skips an
existing chunk only after reopening both partial files and validating graph
IDs, seeds, parameters, dimensions, offsets, and required scalar fields.

Example affine submission:

```bash
export PA_MODEL_FAMILY=affine_linear
export PA_N=5000 PA_M=10 PA_NUM_GRAPHS=500 PA_CHUNK_SIZE=5
export PA_DEGREE_CCDF_ALPHA=2.0
export PA_GRAPH_SEED_BASE=27402
qsub -v PA_PROJECT_ROOT,PA_OUTPUT_ROOT,PA_PYTHON,PA_MODEL_FAMILY,PA_N,PA_M,PA_NUM_GRAPHS,PA_CHUNK_SIZE,PA_DEGREE_CCDF_ALPHA,PA_GRAPH_SEED_BASE tools/hoffman_submit_pa_mobility_edge.sh
```

After all tasks finish, submit `tools/hoffman_aggregate_pa_mobility_edge.sh`
with the same environment. Do not aggregate on the laptop.

For nonlinear production replace `PA_DEGREE_CCDF_ALPHA` by
`PA_ATTACHMENT_ZETA` and set `PA_MODEL_FAMILY=nonlinear`. The nonlinear
kernel exponent is not reported as a fitted degree power-law exponent.

## Exact-degree rewiring

The same chunk and aggregation scripts implement the targeted rewiring
control. Add:

```bash
export PA_GRAPH_VARIANT=rewired
export PA_REWIRE_SEED_BASE=29000
export PA_SWAPS_PER_EDGE=10
export PA_MAX_TRIES_FACTOR=20
```

The output is nested below the selected original parameter directory in
`rewired/`. Every chunk verifies exact labeled-degree equality, records edge
overlap and connectivity, and passes the rewired graph through the identical
full-spectrum cache and post-processing pipeline. After both variants are
aggregated, run
`small_world_mobility_edge.preferential_attachment.mobility_edge_rewiring_comparison`
on the original parameter directory for paired graph/bin differences.

## Controlled family and m/density summaries

`tools/hoffman_pa_family_summary.sh` builds the aligned family figures and
applies the predeclared selection rule for the m-sensitivity target. Pass the
parameter values through `PA_PARAMETER_VALUES` separated by colons, because
SGE uses commas to separate environment assignments. The selected value is
saved in `controlled_family_summary/m_sensitivity_selection.json`.

The m-sensitivity output is explicitly labeled as a density sensitivity:
changing m changes mean degree from approximately 2m. It reuses the selected
m=10 result and varies only m over 2, 5, 10, and 20.

## Cache-only legacy reprocessing

This command uses no eigendecomposition:

```bash
PYTHONPATH=src "$PA_PYTHON" tools/hoffman_pa_cached_reprocess.py \
  --source-root <HOFFMAN_HOME>/small_world_mobility_edge_cache/pa_full_20260813 \
  --provenance-root <HOFFMAN_HOME>/small_world_mobility_edge_cache/pa_full_20260813_provenance/durable_copy \
  --output-root "$PA_OUTPUT_ROOT"
```

## Copy-back

Run locally after remote aggregation:

```bash
export PA_REMOTE_HOST=hoffman
export PA_REMOTE_OUTPUT_ROOT=<HOFFMAN_SCRATCH>/small_world_mobility_edge_pa_mobility_20260823/outputs/preferential_attachment/mobility_edge
export PA_LOCAL_OUTPUT_ROOT=$PWD/outputs/preferential_attachment/mobility_edge
tools/sync_pa_mobility_edge_results.sh
```

This intentionally excludes production `partials/`. Large caches remain on
Hoffman and are referenced by `source_cache.json`; summaries, manifests,
cross-fit models, tables, and plots are copied back.
