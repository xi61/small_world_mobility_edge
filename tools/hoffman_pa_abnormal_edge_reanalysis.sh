#!/bin/bash
#$ -cwd
#$ -N sw_pa_edge_rean
#$ -j y
#$ -o logs/abnormal_edge_reanalysis_$JOB_ID.$TASK_ID.log
#$ -l h_rt=12:00:00
#$ -l h_data=24G
#$ -pe shared 2

set -euo pipefail

: "${USER:?USER must be set}"
HOFFMAN_SCRATCH="${HOFFMAN_SCRATCH:-/u/scratch/${USER:0:1}/${USER}}"
HOFFMAN_HOME="${HOFFMAN_HOME:-/u/home/${USER:0:1}/${USER}}"
REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
: "${PA_PROJECT_ROOT:?set PA_PROJECT_ROOT}"
: "${PA_OUTPUT_ROOT:?set PA_OUTPUT_ROOT}"
: "${PA_FOLLOWUP_OUTPUT_ROOT:?set PA_FOLLOWUP_OUTPUT_ROOT}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ Hoffman module}"
cd "$PA_PROJECT_ROOT"
mkdir -p ${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/logs
export PYTHONPATH=src
export MPLCONFIGDIR=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/.mplconfig
export XDG_CACHE_HOME=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/.cache
export OMP_NUM_THREADS=${NSLOTS:-2}
export OPENBLAS_NUM_THREADS=${NSLOTS:-2}
export MKL_NUM_THREADS=${NSLOTS:-2}
export NUMEXPR_NUM_THREADS=${NSLOTS:-2}
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

case "${SGE_TASK_ID:-1}" in
  1) PA_MODEL_KEY=affine_alpha_2p0 ;;
  2) PA_MODEL_KEY=affine_alpha_1p25 ;;
  3) PA_MODEL_KEY=affine_alpha_1p05 ;;
  4) PA_MODEL_KEY=superlinear_zeta_1p5 ;;
  *) echo "invalid task ${SGE_TASK_ID:-}" >&2; exit 2 ;;
esac

python3 tools/hoffman_pa_abnormal_edge_reanalysis.py \
  --source-root "$PA_OUTPUT_ROOT" \
  --output-root "$PA_FOLLOWUP_OUTPUT_ROOT" \
  --model-key "$PA_MODEL_KEY"
