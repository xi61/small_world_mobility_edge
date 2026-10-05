#!/bin/bash
#$ -cwd
#$ -N rs_aggregate
#$ -j y
#$ -o logs/aggregate_$JOB_ID.log
#$ -l h_rt=00:30:00
#$ -l h_data=4G
#$ -pe shared 1

set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${HOFFMAN_HOME:?set HOFFMAN_HOME to your scheduler home root}"

: "${RS_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${RS_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_20260921}"
: "${RS_REPORT_DIR:=$RS_ROOT/outputs/report}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$RS_PROJECT_ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python3 tools/hoffman_random_schrodinger_aggregate.py \
  --root "$RS_ROOT" \
  --output-dir "$RS_REPORT_DIR"
