#!/bin/bash
#$ -cwd
#$ -N lpa2_report_v2
#$ -j y
#$ -o logs/report_$JOB_ID.log
#$ -l h_rt=04:00:00
#$ -l h_data=16G
#$ -pe shared 1
set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${HOFFMAN_HOME:?set HOFFMAN_HOME to your scheduler home root}"
: "${LP_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${LP_SCIENCE_ROOT:=${HOFFMAN_SCRATCH}/linear_pa_two_part_20260924_v3}"
: "${LP_REPORT_RUN_ROOT:=${HOFFMAN_SCRATCH}/linear_pa_two_part_20260924_v3_reporting_v5}"
: "${LP_CONFIG:=$LP_PROJECT_ROOT/experiments/linear_pa_two_part_20260924/config.json}"
: "${LP_FREEZE_MANIFEST:=$LP_REPORT_RUN_ROOT/freeze/executed_reporting_source_v5/freeze_manifest.json}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$LP_PROJECT_ROOT"
export PYTHONPATH=src
export MPLCONFIGDIR="$LP_REPORT_RUN_ROOT/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 tools/hoffman_linear_pa_two_part.py \
  --project-root "$LP_PROJECT_ROOT" \
  --config "$LP_CONFIG" \
  --freeze-manifest "$LP_FREEZE_MANIFEST" \
  aggregate \
  --output-root "$LP_SCIENCE_ROOT/cases" \
  --report-dir "$LP_REPORT_RUN_ROOT/report"
