#!/bin/bash
#$ -cwd
#$ -N rs_corr_s1
#$ -j y
#$ -o logs/stage1_$JOB_ID.$TASK_ID.log
#$ -l h_rt=01:00:00
#$ -l h_data=4G
#$ -pe shared 1
#$ -t 1-3

set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${RS_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${RS_SOURCE_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_single_graph_20260901/outputs}"
: "${RS_OLD_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_20260921}"
: "${RS_CORRECTION_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_correction_20260922_v1}"
: "${RS_FREEZE_MANIFEST:=$RS_CORRECTION_ROOT/freeze/executed_correction_source/freeze_manifest.json}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$RS_PROJECT_ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$RS_CORRECTION_ROOT/.mplconfig" XDG_CACHE_HOME="$RS_CORRECTION_ROOT/.cache"
mkdir -p "$RS_CORRECTION_ROOT/logs" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
case "${SGE_TASK_ID:-}" in
  1) MODEL=affine_alpha_1p05 ;;
  2) MODEL=nonlinear_zeta_1p5 ;;
  3) MODEL=affine_alpha_2p0 ;;
  *) echo "invalid SGE_TASK_ID=${SGE_TASK_ID:-unset}" >&2; exit 2 ;;
esac
python3 tools/hoffman_random_schrodinger_corrections.py \
  --project-root "$RS_PROJECT_ROOT" --freeze-manifest "$RS_FREEZE_MANIFEST" stage1 \
  --source-dir "$RS_SOURCE_ROOT/$MODEL" \
  --original-stage1-dir "$RS_OLD_ROOT/stage1/$MODEL" \
  --output-dir "$RS_CORRECTION_ROOT/stage1/$MODEL"
