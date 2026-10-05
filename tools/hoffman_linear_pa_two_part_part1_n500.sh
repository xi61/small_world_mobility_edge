#!/bin/bash
#$ -cwd
#$ -N lpa2v3_p1_500
#$ -j y
#$ -o logs/part1_N500_$JOB_ID.$TASK_ID.log
#$ -l h_rt=06:00:00
#$ -l h_data=8G
#$ -pe shared 2
set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${HOFFMAN_HOME:?set HOFFMAN_HOME to your scheduler home root}"
: "${LP_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${LP_RUN_ROOT:=${HOFFMAN_SCRATCH}/linear_pa_two_part_20260924_v3}"
: "${LP_CONFIG:=$LP_PROJECT_ROOT/experiments/linear_pa_two_part_20260924/config.json}"
: "${LP_FREEZE_MANIFEST:=$LP_RUN_ROOT/freeze/executed_source_v3/freeze_manifest.json}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$LP_PROJECT_ROOT"
export PYTHONPATH=src
PILOT_ALPHA_SLUG=$(python3 -c 'from small_world_mobility_edge.preferential_attachment.linear_pa_two_part import alpha_slug; print(alpha_slug(2.0))')
python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("status")=="COMPLETE" and int(x.get("dictionary_mode_row_count", 0)) > 0' "$LP_RUN_ROOT/pilot/N_80/alpha_${PILOT_ALPHA_SLUG}/part2/manifest.json"
export OMP_NUM_THREADS=${NSLOTS:-2} OPENBLAS_NUM_THREADS=${NSLOTS:-2} MKL_NUM_THREADS=${NSLOTS:-2} NUMEXPR_NUM_THREADS=${NSLOTS:-2}
ALPHA=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["alpha_grid"][int(sys.argv[2])-1])' "$LP_CONFIG" "${SGE_TASK_ID:?}")
python3 tools/hoffman_linear_pa_two_part.py --project-root "$LP_PROJECT_ROOT" --config "$LP_CONFIG" --freeze-manifest "$LP_FREEZE_MANIFEST" part1 --n 500 --alpha "$ALPHA" --output-root "$LP_RUN_ROOT/cases"
