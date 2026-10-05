#!/bin/bash
#$ -cwd
#$ -N lpa2v3_p2_10k
#$ -j y
#$ -o logs/part2_N10000_$JOB_ID.$TASK_ID.log
#$ -l h_rt=24:00:00
#$ -l h_data=16G
#$ -pe shared 4
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
export OMP_NUM_THREADS=${NSLOTS:-4} OPENBLAS_NUM_THREADS=${NSLOTS:-4} MKL_NUM_THREADS=${NSLOTS:-4} NUMEXPR_NUM_THREADS=${NSLOTS:-4}
ALPHA=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["alpha_grid"][int(sys.argv[2])-1])' "$LP_CONFIG" "${SGE_TASK_ID:?}")
python3 tools/hoffman_linear_pa_two_part.py --project-root "$LP_PROJECT_ROOT" --config "$LP_CONFIG" --freeze-manifest "$LP_FREEZE_MANIFEST" part2 --n 10000 --alpha "$ALPHA" --output-root "$LP_RUN_ROOT/cases"
