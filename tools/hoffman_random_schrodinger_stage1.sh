#!/bin/bash
#$ -cwd
#$ -N rs_stage1
#$ -t 1-3
#$ -j y
#$ -o logs/stage1_$JOB_ID.$TASK_ID.log
#$ -l h_rt=02:00:00
#$ -l h_data=4G
#$ -pe shared 1

set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${HOFFMAN_HOME:?set HOFFMAN_HOME to your scheduler home root}"

: "${RS_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${RS_SOURCE_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_single_graph_20260901/outputs}"
: "${RS_OUTPUT_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_20260921/stage1}"
: "${RS_BASELINE_SAMPLES:=100}"

source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$RS_PROJECT_ROOT"
export PYTHONPATH=src
export MPLCONFIGDIR=${HOFFMAN_SCRATCH}/random_schrodinger_20260921/.mplconfig_stage1
export XDG_CACHE_HOME=${HOFFMAN_SCRATCH}/random_schrodinger_20260921/.cache_stage1
export OMP_NUM_THREADS=${NSLOTS:-1}
export OPENBLAS_NUM_THREADS=${NSLOTS:-1}
export MKL_NUM_THREADS=${NSLOTS:-1}
export NUMEXPR_NUM_THREADS=${NSLOTS:-1}
mkdir -p "$RS_OUTPUT_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" ${HOFFMAN_SCRATCH}/random_schrodinger_20260921/logs

case "${SGE_TASK_ID:-}" in
  1) MODEL_KEY=affine_alpha_1p05 ;;
  2) MODEL_KEY=nonlinear_zeta_1p5 ;;
  3) MODEL_KEY=affine_alpha_2p0 ;;
  *) echo "invalid SGE_TASK_ID=${SGE_TASK_ID:-unset}" >&2; exit 2 ;;
esac

python3 tools/hoffman_random_schrodinger.py stage1 \
  --source-dir "$RS_SOURCE_ROOT/$MODEL_KEY" \
  --output-dir "$RS_OUTPUT_ROOT/$MODEL_KEY" \
  --project-root "$RS_PROJECT_ROOT" \
  --baseline-samples "$RS_BASELINE_SAMPLES" \
  --baseline-seed 945101
