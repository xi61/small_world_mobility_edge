#!/bin/bash
#$ -cwd
#$ -N pa_cp_test
#$ -j y
#$ -o logs/tests_$JOB_ID.log
#$ -l h_rt=00:30:00
#$ -l h_data=4G
#$ -pe shared 1
set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${RS_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${RS_HANDOFF_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_pa_coupling_and_anderson_20260924_v1}"
: "${RS_FREEZE_MANIFEST:=$RS_HANDOFF_ROOT/freeze/executed_clean_pa_source_v6/freeze_manifest.json}"
: "${RS_CONFIG:=$RS_PROJECT_ROOT/experiments/pa_coupling_and_anderson_20260924/clean_pa_execution_config.json}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$RS_PROJECT_ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$RS_HANDOFF_ROOT/logs" "$RS_HANDOFF_ROOT/validation"
python3 -m compileall -q src/small_world_mobility_edge/random_schrodinger tools/hoffman_pa_coupling_handoff.py
python3 -m unittest -v tests.test_pa_coupling_handoff
python3 tools/hoffman_pa_coupling_handoff.py --project-root "$RS_PROJECT_ROOT" --freeze-manifest "$RS_FREEZE_MANIFEST" --config "$RS_CONFIG" validate --output-dir "$RS_HANDOFF_ROOT/validation"
