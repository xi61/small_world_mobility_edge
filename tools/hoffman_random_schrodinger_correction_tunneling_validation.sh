#!/bin/bash
#$ -cwd
#$ -N rs_corr_tval
#$ -j y
#$ -o logs/tunneling_validation_$JOB_ID.log
#$ -l h_rt=00:30:00
#$ -l h_data=4G
#$ -pe shared 1

set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${RS_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${RS_CORRECTION_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_correction_20260922_v1}"
: "${RS_FREEZE_MANIFEST:=$RS_CORRECTION_ROOT/freeze/executed_correction_source/freeze_manifest.json}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$RS_PROJECT_ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$RS_CORRECTION_ROOT/logs"
python3 tools/hoffman_random_schrodinger_corrections.py \
  --project-root "$RS_PROJECT_ROOT" --freeze-manifest "$RS_FREEZE_MANIFEST" tunneling-validation \
  --output-dir "$RS_CORRECTION_ROOT/tunneling_validation"
