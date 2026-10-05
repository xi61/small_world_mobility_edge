#!/bin/bash
#$ -cwd
#$ -N rs_audit_verify
#$ -j y
#$ -o logs/audit_verify_$JOB_ID.log
#$ -l h_rt=00:20:00
#$ -l h_data=4G
#$ -pe shared 1

set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"

: "${RS_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${RS_PRODUCTION_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_20260921}"
: "${RS_AUDIT_OUTPUT:=${HOFFMAN_SCRATCH}/random_schrodinger_20260921/audit_20260922}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$RS_PROJECT_ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$RS_AUDIT_OUTPUT/logs" "$RS_AUDIT_OUTPUT/evidence"
python3 tools/hoffman_random_schrodinger_audit_verify.py \
  --root "$RS_PRODUCTION_ROOT" \
  --output-dir "$RS_AUDIT_OUTPUT/evidence"
