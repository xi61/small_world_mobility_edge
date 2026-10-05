#!/bin/bash
#$ -cwd
#$ -N rs_corr_tests
#$ -j y
#$ -o logs/tests_$JOB_ID.log
#$ -l h_rt=00:20:00
#$ -l h_data=4G
#$ -pe shared 1

set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${RS_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${RS_CORRECTION_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_correction_20260922_v1}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$RS_PROJECT_ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$RS_CORRECTION_ROOT/logs"
python3 -m compileall -q src/small_world_mobility_edge/random_schrodinger tools/hoffman_random_schrodinger_corrections.py
python3 -m unittest discover -v -s tests -p 'test_random_schrodinger_*.py'
