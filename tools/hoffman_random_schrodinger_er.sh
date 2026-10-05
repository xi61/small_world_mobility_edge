#!/bin/bash
#$ -cwd
#$ -N rs_er
#$ -j y
#$ -o logs/er_$JOB_ID.log
#$ -l h_rt=08:00:00
#$ -l h_data=8G
#$ -pe shared 2

set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${HOFFMAN_HOME:?set HOFFMAN_HOME to your scheduler home root}"

: "${RS_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${RS_OUTPUT_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_20260921/outputs}"
: "${RS_ER_N:=1000}"
: "${RS_ER_SEED:=21901001}"
: "${RS_ER_LABEL:=pilot}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$RS_PROJECT_ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=${NSLOTS:-2}
export OPENBLAS_NUM_THREADS=${NSLOTS:-2}
export MKL_NUM_THREADS=${NSLOTS:-2}
export NUMEXPR_NUM_THREADS=${NSLOTS:-2}
export MPLCONFIGDIR=${HOFFMAN_SCRATCH}/random_schrodinger_20260921/.mplconfig
export XDG_CACHE_HOME=${HOFFMAN_SCRATCH}/random_schrodinger_20260921/.cache
mkdir -p "$RS_OUTPUT_ROOT/er_benchmark" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python3 tools/hoffman_random_schrodinger.py er-benchmark \
  --n "$RS_ER_N" \
  --graph-seed "$RS_ER_SEED" \
  --run-label "$RS_ER_LABEL" \
  --output-dir "$RS_OUTPUT_ROOT/er_benchmark/${RS_ER_LABEL}_N${RS_ER_N}_seed${RS_ER_SEED}" \
  --project-root "$RS_PROJECT_ROOT"
