#!/bin/bash
#$ -cwd
#$ -N pa_cp_agg
#$ -j y
#$ -o logs/aggregate_$JOB_ID.log
#$ -l h_rt=00:45:00
#$ -l h_data=6G
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
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$RS_HANDOFF_ROOT/logs"
for model in affine_alpha_1p05 affine_alpha_2p0 nonlinear_zeta_1p5; do
  python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("status")=="COMPLETE"' "$RS_HANDOFF_ROOT/clean_pa/$model/run_manifest.json"
done
python3 tools/hoffman_pa_coupling_handoff.py --project-root "$RS_PROJECT_ROOT" --freeze-manifest "$RS_FREEZE_MANIFEST" --config "$RS_CONFIG" aggregate --clean-pa-root "$RS_HANDOFF_ROOT/clean_pa" --output-dir "$RS_HANDOFF_ROOT/aggregate_v2"
