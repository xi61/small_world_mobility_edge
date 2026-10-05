#!/bin/bash
#$ -cwd
#$ -N lpa2v3_pilot
#$ -j y
#$ -o logs/pilot_$JOB_ID.log
#$ -l h_rt=02:00:00
#$ -l h_data=6G
#$ -pe shared 1
set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${HOFFMAN_HOME:?set HOFFMAN_HOME to your scheduler home root}"
: "${LP_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${LP_RUN_ROOT:=${HOFFMAN_SCRATCH}/linear_pa_two_part_20260924_v3}"
: "${LP_FREEZE_MANIFEST:=$LP_RUN_ROOT/freeze/executed_source_v3/freeze_manifest.json}"
: "${LP_PILOT_CONFIG:=$LP_PROJECT_ROOT/experiments/linear_pa_two_part_20260924/pilot_config.json}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$LP_PROJECT_ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("status")=="COMPLETE" and x.get("validation_passed") is True' "$LP_RUN_ROOT/validation/run_manifest.json"
python3 tools/hoffman_linear_pa_two_part.py --project-root "$LP_PROJECT_ROOT" --config "$LP_PILOT_CONFIG" --freeze-manifest "$LP_FREEZE_MANIFEST" part1 --n 80 --alpha 2.0 --output-root "$LP_RUN_ROOT/pilot"
python3 tools/hoffman_linear_pa_two_part.py --project-root "$LP_PROJECT_ROOT" --config "$LP_PILOT_CONFIG" --freeze-manifest "$LP_FREEZE_MANIFEST" part2 --n 80 --alpha 2.0 --output-root "$LP_RUN_ROOT/pilot"
PILOT_ALPHA_SLUG=$(python3 -c 'from small_world_mobility_edge.preferential_attachment.linear_pa_two_part import alpha_slug; print(alpha_slug(2.0))')
python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("status")=="COMPLETE" and int(x.get("dictionary_mode_row_count", 0)) > 0 and int(x.get("raw_local_eigensystem_cache_count", 0)) > 0' "$LP_RUN_ROOT/pilot/N_80/alpha_${PILOT_ALPHA_SLUG}/part2/manifest.json"
