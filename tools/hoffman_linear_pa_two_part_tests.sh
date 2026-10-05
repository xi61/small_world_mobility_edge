#!/bin/bash
#$ -cwd
#$ -N lpa2v3_tests
#$ -j y
#$ -o logs/tests_$JOB_ID.log
#$ -l h_rt=01:00:00
#$ -l h_data=6G
#$ -pe shared 1
set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${HOFFMAN_HOME:?set HOFFMAN_HOME to your scheduler home root}"
: "${LP_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${LP_RUN_ROOT:=${HOFFMAN_SCRATCH}/linear_pa_two_part_20260924_v3}"
: "${LP_FREEZE_MANIFEST:=$LP_RUN_ROOT/freeze/executed_source_v3/freeze_manifest.json}"
: "${LP_CONFIG:=$LP_PROJECT_ROOT/experiments/linear_pa_two_part_20260924/config.json}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$LP_PROJECT_ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$LP_RUN_ROOT/logs" "$LP_RUN_ROOT/validation"
python3 -m compileall -q src/small_world_mobility_edge/preferential_attachment tools/hoffman_linear_pa_two_part.py
python3 -m unittest -v tests.test_linear_pa_two_part
python3 tools/hoffman_linear_pa_two_part.py --project-root "$LP_PROJECT_ROOT" --config "$LP_CONFIG" --freeze-manifest "$LP_FREEZE_MANIFEST" validate --output-dir "$LP_RUN_ROOT/validation"
