#!/bin/bash
#$ -cwd
#$ -N and_pa_rep
#$ -j y
#$ -o logs/pa_$JOB_ID.$TASK_ID.log
#$ -l h_rt=12:00:00
#$ -l h_data=8G
#$ -pe shared 2
#$ -t 1-27
#$ -tc 4
set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${RS_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${RS_RUN_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_repair_and_extension_20260924_v1}"
: "${RS_CONFIG:=$RS_PROJECT_ROOT/experiments/random_schrodinger/anderson_repair_extension_20260924_v1.json}"
: "${RS_FREEZE_MANIFEST:=$RS_RUN_ROOT/freezes/executed_anderson_source_v1/freeze_manifest.json}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$RS_PROJECT_ROOT"
export PYTHONPATH=src MPLBACKEND=Agg
export OMP_NUM_THREADS=${NSLOTS:-2} OPENBLAS_NUM_THREADS=${NSLOTS:-2} MKL_NUM_THREADS=${NSLOTS:-2} NUMEXPR_NUM_THREADS=${NSLOTS:-2}
python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("validation_passed") is True' "$RS_RUN_ROOT/validation/run_manifest.json"
python3 tools/hoffman_anderson_execution.py --project-root "$RS_PROJECT_ROOT" --config "$RS_CONFIG" --freeze-manifest "$RS_FREEZE_MANIFEST" pa-replication --task-id "${SGE_TASK_ID:?}" --output-root "$RS_RUN_ROOT/fresh_pa"
