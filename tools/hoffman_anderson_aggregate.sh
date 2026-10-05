#!/bin/bash
#$ -cwd
#$ -N and_agg
#$ -j y
#$ -o logs/aggregate_$JOB_ID.log
#$ -l h_rt=04:00:00
#$ -l h_data=16G
#$ -pe shared 1
set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${RS_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${RS_RUN_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_repair_and_extension_20260924_v1}"
: "${RS_CONFIG:=$RS_PROJECT_ROOT/experiments/random_schrodinger/anderson_repair_extension_20260924_v1.json}"
: "${RS_FREEZE_MANIFEST:=$RS_RUN_ROOT/freezes/executed_anderson_source_v1/freeze_manifest.json}"
: "${RS_AGGREGATE_PHASE:=final}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$RS_PROJECT_ROOT"
export PYTHONPATH=src MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ "$RS_AGGREGATE_PHASE" == "pre_a9" ]]; then
  OUTPUT_DIR="$RS_RUN_ROOT/aggregate_pre_a9"
else
  OUTPUT_DIR="$RS_RUN_ROOT/aggregate"
fi
python3 tools/hoffman_anderson_execution.py --project-root "$RS_PROJECT_ROOT" --config "$RS_CONFIG" --freeze-manifest "$RS_FREEZE_MANIFEST" aggregate --run-root "$RS_RUN_ROOT" --output-dir "$OUTPUT_DIR" --phase "$RS_AGGREGATE_PHASE"
