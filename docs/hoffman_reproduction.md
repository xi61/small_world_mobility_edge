# Hoffman reproduction guide

The public repository contains the local scientific modules and the Hoffman
SGE/Grid Engine wrappers used to split, aggregate, copy back, and validate the
production runs. The wrappers are retained as execution provenance; historical
scripts that contain a dated scratch path should be adapted to the new clone or
supplied with the documented environment variables before resubmission.

## Environment

On a Hoffman login node, clone the repository into project or scratch storage,
load a Python version compatible with `pyproject.toml`, and install the package:

```bash
git clone https://github.com/xi61/small_world_mobility_edge.git
cd small_world_mobility_edge
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
mkdir -p logs
export PROJECT_ROOT="$PWD"
export HOFFMAN_SCRATCH="/u/scratch/${USER:0:1}/$USER"
export HOFFMAN_HOME="/u/home/${USER:0:1}/$USER"
# Choose an available Hoffman module that provides Python 3.11 or newer.
export PYTHON_MODULE="python/<version>"  # replace using `module avail python`
export PYTHONPATH="$PROJECT_ROOT/src"
```

The exact NumPy, SciPy, NetworkX, Python, and seed metadata recorded by an old
run remains in its manifest. Reproducing an historical result as closely as
possible means using those recorded versions; validating the current public
code means using the package requirements and test suite.

## Model and pipeline map

| Family | Local/scientific entry point | Hoffman chunk or submit path | Aggregation/validation |
|---|---|---|---|
| Random `k`-regular | `python3 -m small_world_mobility_edge.model_smoke --model regular` and `regular_baseline_experiments.py` | No separate transferred production-array wrapper was found | Output invariants are written by `model_smoke.py` |
| Erdős--Rényi | `erdos_renyi_spacing_by_energy.py`, `erdos_renyi_extreme_edge.py` | Run the modules directly or wrap their CLI in an SGE array | Manifests and summaries are emitted by the modules |
| Chung--Lu percentile/localization | `chung_lu_eigenvector_percentile_scan.py` | `hoffman_submit_chung_lu_eigenvector_percentile.sh` plus `hoffman_chung_lu_eigenvector_percentile_chunk.py` | `hoffman_chung_lu_eigenvector_percentile_aggregate.py` |
| Chung--Lu resonance/LCC/projector | resonance, LCC, and projector modules under `src/` | `hoffman_submit_chung_lu_eigenvalue_resonance.sh`, `hoffman_submit_chung_lu_eigenvector_lcc.sh`, `hoffman_submit_chung_lu_e1_projector.sh` | matching `*_aggregate.py` utilities and `audit_chung_lu_e1_resonance_structure.py` |
| Chung--Lu joint spacing--IPR | `chung_lu_joint_spacing_ipr.py` | `hoffman_submit_chung_lu_joint_spacing_ipr.sh` and `hoffman_chung_lu_joint_spacing_ipr_chunk.py` | `hoffman_chung_lu_joint_spacing_ipr_aggregate.py` |
| Fixed realized-degree control | `fixed_degree_configuration_joint_tail.py` | `hoffman_fixed_degree_configuration_pair.sh` | `hoffman_fixed_degree_configuration_aggregate.sh` |
| PA full spectrum | `preferential_attachment/full_spectrum_scan.py` | `hoffman_submit_preferential_attachment_full_spectrum.sh` and `hoffman_preferential_attachment_full_spectrum_chunk.py` | `hoffman_aggregate_preferential_attachment_full_spectrum.sh` and the matching Python aggregator |
| PA mobility-edge production | `preferential_attachment/full_spectrum_scan.py` and `mobility_edge_postprocess.py` | `hoffman_submit_pa_mobility_edge.sh` and `hoffman_pa_mobility_edge_chunk.py` | `hoffman_aggregate_pa_mobility_edge.sh`, `hoffman_pa_mobility_edge_aggregate.py`, family and sensitivity summaries |
| PA rewiring/structure | rewiring and production-aggregate modules | production chunks plus `hoffman_aggregate_pa_structural_replay.sh` | `hoffman_pa_structural_replay_aggregate.py`, `hoffman_pa_rewiring_comparison.sh` |
| PA extreme/abnormal edge | extreme-edge and abnormal-edge modules | `hoffman_pa_extreme_edge_augmentation.sh`, `hoffman_pa_abnormal_edge_sparse.sh`, `hoffman_pa_abnormal_edge_reanalysis.sh`, `hoffman_pa_abnormal_band_microscope.sh` | matching aggregate, validation, and report builders under `tools/` |
| PA single-graph edge/bulk | `preferential_attachment/single_graph_edge_bulk.py` | `hoffman_pa_single_graph_edge_bulk.sh` and `.py` | `summarize_pa_single_graph_edge_bulk.py` and `build_pa_single_graph_pdf_report.py` |
| Clean affine-PA two-part | `preferential_attachment/linear_pa_two_part.py` | `hoffman_submit_linear_pa_two_part_pipeline.sh` and part-specific wrappers | `hoffman_linear_pa_two_part_aggregate.sh`, `linear_pa_two_part_reporting.py` |
| PA diagonal uniform noise | `random_schrodinger/pa_diagonal_uniform_noise.py` | freeze plus `hoffman_pa_diagonal_uniform_noise.py` operator jobs | `pa_diagonal_uniform_noise_reporting.py` case/global aggregation |
| Random 3-regular Anderson | `random_schrodinger/anderson_rrg_benchmark.py` | `hoffman_anderson_rrg.sh`, `hoffman_anderson_rrg_a9.sh` | `hoffman_anderson_aggregate.sh`, `anderson_reporting.py` |
| Geometry/tunneling corrections | `random_schrodinger/geometry.py`, `tunneling_correction.py` | `hoffman_random_schrodinger_correction_*.sh` | `hoffman_random_schrodinger_corrections.py` and aggregate wrapper |
| Cached PA multifractality | `preferential_attachment/multifractality.py` | `hoffman_submit_pa_multifractality.sh` | `hoffman_pa_multifractality.py aggregate`; [exact environment and cache schema](experiments/pa_multifractality.md) |

## PA production example

The primary PA array wrapper reads its scientific settings from environment
variables. A typical affine submission is:

```bash
export PA_PROJECT_ROOT="$PROJECT_ROOT"
export PA_OUTPUT_ROOT="$PROJECT_ROOT/outputs/preferential_attachment/mobility_edge"
export PA_MODEL_FAMILY=affine_linear
export PA_N=5000 PA_M=10 PA_NUM_GRAPHS=500 PA_CHUNK_SIZE=5
export PA_DEGREE_CCDF_ALPHA=2.0
export PA_GRAPH_SEED_BASE=27402
qsub -v PA_PROJECT_ROOT,PA_OUTPUT_ROOT,PA_MODEL_FAMILY,PA_N,PA_M,PA_NUM_GRAPHS,PA_CHUNK_SIZE,PA_DEGREE_CCDF_ALPHA,PA_GRAPH_SEED_BASE tools/hoffman_submit_pa_mobility_edge.sh
```

For nonlinear PA, replace `PA_DEGREE_CCDF_ALPHA` with
`PA_ATTACHMENT_ZETA` and set `PA_MODEL_FAMILY=nonlinear`. Chunk validation
checks graph indices, seeds, parameters, dimensions, and required fields before
accepting an existing checkpoint.

## Copy-back and verification

The `sync_pa_*.sh` scripts use `rsync` and environment variables for remote and
local roots. They exclude or isolate restart shards as documented by each
script. After transfer, run the matching audit or validation utility before
aggregation. Do not treat filename existence alone as completion.

The Git publication tier contains final plots, summary CSVs, manifests, and
reports. Per-graph compressed tables and `.npz` checkpoints are regenerated
artifacts and stay ignored; see `docs/results_inventory.md` for the precise
boundary.
