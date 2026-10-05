#!/bin/bash
#$ -cwd
#$ -N and_a0_supp
#$ -j y
#$ -o logs/supplement_$JOB_ID.$TASK_ID.log
#$ -l h_rt=08:00:00
#$ -l h_data=8G
#$ -pe shared 2
#$ -t 1-3
#$ -tc 1
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
python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("status")=="COMPLETE"' "$RS_RUN_ROOT/pilot_postprocessing/affine_alpha_1p05/potential_seed_931701/run_manifest.json"
MODELS=(affine_alpha_1p05 nonlinear_zeta_1p5 affine_alpha_2p0)
MODEL=${MODELS[$((SGE_TASK_ID - 1))]}
python3 tools/hoffman_anderson_execution.py --project-root "$RS_PROJECT_ROOT" --config "$RS_CONFIG" --freeze-manifest "$RS_FREEZE_MANIFEST" baseline-supplement --model-key "$MODEL" --output-dir "$RS_RUN_ROOT/targeted_baseline_supplements/$MODEL"
