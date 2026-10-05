#!/bin/bash
#$ -cwd
#$ -N rs_pa_disorder
#$ -j y
#$ -o logs/disorder_$JOB_ID.$TASK_ID.log
#$ -l h_rt=08:00:00
#$ -l h_data=8G
#$ -pe shared 2
#$ -t 1-6
#$ -tc 2

set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${HOFFMAN_HOME:?set HOFFMAN_HOME to your scheduler home root}"

: "${RS_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${RS_SOURCE_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_single_graph_20260901/outputs}"
: "${RS_OUTPUT_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_20260921/outputs}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$RS_PROJECT_ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=${NSLOTS:-2}
export OPENBLAS_NUM_THREADS=${NSLOTS:-2}
export MKL_NUM_THREADS=${NSLOTS:-2}
export NUMEXPR_NUM_THREADS=${NSLOTS:-2}
export MPLCONFIGDIR=${HOFFMAN_SCRATCH}/random_schrodinger_20260921/.mplconfig
export XDG_CACHE_HOME=${HOFFMAN_SCRATCH}/random_schrodinger_20260921/.cache

STAGE2_MANIFESTS=(
  "$RS_OUTPUT_ROOT/tunneling_validation/run_manifest.json"
  "$RS_OUTPUT_ROOT/er_benchmark/published_size_N10000_seed21901001/run_manifest.json"
  "$RS_OUTPUT_ROOT/tunneling_applications/er/run_manifest.json"
  "$RS_OUTPUT_ROOT/tunneling_applications/pa/run_manifest.json"
  "$RS_OUTPUT_ROOT/tunneling_applications/cl/run_manifest.json"
)
for manifest in "${STAGE2_MANIFESTS[@]}"; do
  python3 -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); assert d.get("status") == "COMPLETE", (p,d.get("status"))' "$manifest"
done

MODEL_INDEX=$(( (SGE_TASK_ID - 1) / 2 ))
POTENTIAL_INDEX=$(( (SGE_TASK_ID - 1) % 2 ))
case "$MODEL_INDEX" in
  0) MODEL=affine_alpha_1p05 ;;
  1) MODEL=nonlinear_zeta_1p5 ;;
  2) MODEL=affine_alpha_2p0 ;;
  *) echo "unsupported model index $MODEL_INDEX" >&2; exit 2 ;;
esac
POTENTIAL_SEEDS=(931701 931702)
POTENTIAL_SEED=${POTENTIAL_SEEDS[$POTENTIAL_INDEX]}
OUTPUT_DIR="$RS_OUTPUT_ROOT/pa_disorder_pilot/$MODEL/potential_seed_$POTENTIAL_SEED"
mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python3 tools/hoffman_random_schrodinger.py pa-disorder \
  --source-dir "$RS_SOURCE_ROOT/$MODEL" \
  --potential-seed "$POTENTIAL_SEED" \
  --output-dir "$OUTPUT_DIR" \
  --project-root "$RS_PROJECT_ROOT"
