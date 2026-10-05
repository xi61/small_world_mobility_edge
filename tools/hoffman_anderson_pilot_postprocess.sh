#!/bin/bash
#$ -cwd
#$ -N and_pilot_pp
#$ -j y
#$ -o logs/pilot_$JOB_ID.$TASK_ID.log
#$ -l h_rt=02:00:00
#$ -l h_data=6G
#$ -pe shared 1
#$ -t 1-6
#$ -tc 3
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("validation_passed") is True' "$RS_RUN_ROOT/validation/run_manifest.json"
MODEL_INDEX=$(( (SGE_TASK_ID - 1) / 2 ))
POTENTIAL_INDEX=$(( (SGE_TASK_ID - 1) % 2 ))
MODELS=(affine_alpha_1p05 nonlinear_zeta_1p5 affine_alpha_2p0)
SEEDS=(931701 931702)
MODEL=${MODELS[$MODEL_INDEX]}
SEED=${SEEDS[$POTENTIAL_INDEX]}
python3 tools/hoffman_anderson_execution.py --project-root "$RS_PROJECT_ROOT" --config "$RS_CONFIG" --freeze-manifest "$RS_FREEZE_MANIFEST" pilot-postprocess --model-key "$MODEL" --potential-seed "$SEED" --output-dir "$RS_RUN_ROOT/pilot_postprocessing/$MODEL/potential_seed_$SEED"
