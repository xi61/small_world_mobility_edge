#!/bin/bash
#$ -cwd
#$ -N sw_pa_single_g
#$ -t 1-3
#$ -j y
#$ -o logs/single_graph_$JOB_ID.$TASK_ID.log
#$ -l h_rt=12:00:00
#$ -l h_data=8G
#$ -pe shared 4

set -euo pipefail

: "${USER:?USER must be set}"
HOFFMAN_SCRATCH="${HOFFMAN_SCRATCH:-/u/scratch/${USER:0:1}/${USER}}"
HOFFMAN_HOME="${HOFFMAN_HOME:-/u/home/${USER:0:1}/${USER}}"
REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$PWD}}"

: "${PA_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${PA_SINGLE_OUTPUT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_single_graph_20260901/outputs}"
: "${PA_REFERENCE_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/outputs/preferential_attachment/mobility_edge}"

source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ Hoffman module}"
cd "$PA_PROJECT_ROOT"
export PYTHONPATH=src
export MPLCONFIGDIR=${HOFFMAN_SCRATCH}/small_world_mobility_edge_single_graph_20260901/.mplconfig
export XDG_CACHE_HOME=${HOFFMAN_SCRATCH}/small_world_mobility_edge_single_graph_20260901/.cache
export OMP_NUM_THREADS=${NSLOTS:-4}
export OPENBLAS_NUM_THREADS=${NSLOTS:-4}
export MKL_NUM_THREADS=${NSLOTS:-4}
export NUMEXPR_NUM_THREADS=${NSLOTS:-4}
mkdir -p "$PA_SINGLE_OUTPUT_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

case "${SGE_TASK_ID:-}" in
  1)
    MODEL_FAMILY=affine_linear
    PARAMETER_VALUE=1.05
    GRAPH_SEED=4011773341
    REFERENCE_CDF="$PA_REFERENCE_ROOT/affine_linear/n_5000_m_10/degree_ccdf_alpha_1p05/unfolding/crossfit_k5/fold_0.npz"
    ;;
  2)
    MODEL_FAMILY=nonlinear
    PARAMETER_VALUE=1.50
    GRAPH_SEED=1912200482
    REFERENCE_CDF="$PA_REFERENCE_ROOT/nonlinear/n_5000_m_10/attachment_zeta_1p5/unfolding/crossfit_k5/fold_0.npz"
    ;;
  3)
    MODEL_FAMILY=affine_linear
    PARAMETER_VALUE=2.00
    GRAPH_SEED=3479052237
    REFERENCE_CDF="$PA_REFERENCE_ROOT/affine_linear/n_5000_m_10/degree_ccdf_alpha_2p0/unfolding/crossfit_k5/fold_0.npz"
    ;;
  *)
    echo "invalid SGE_TASK_ID=${SGE_TASK_ID:-unset}" >&2
    exit 2
    ;;
esac

test -r "$REFERENCE_CDF"
python3 tools/hoffman_pa_single_graph_edge_bulk.py \
  --n 5000 \
  --m 10 \
  --model-family "$MODEL_FAMILY" \
  --parameter-value "$PARAMETER_VALUE" \
  --graph-seed "$GRAPH_SEED" \
  --graph-index 0 \
  --reference-cdf "$REFERENCE_CDF" \
  --output-root "$PA_SINGLE_OUTPUT_ROOT" \
  --histogram-window-width 0.05 \
  --random-baseline-seed 945001 \
  --random-baseline-samples 1000
