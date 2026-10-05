#!/bin/bash
#$ -cwd
#$ -N rs_tunnel_app
#$ -j y
#$ -o logs/tunnel_app_$JOB_ID.$TASK_ID.log
#$ -l h_rt=04:00:00
#$ -l h_data=8G
#$ -pe shared 1
#$ -t 1-3

set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${HOFFMAN_HOME:?set HOFFMAN_HOME to your scheduler home root}"

: "${RS_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${RS_SOURCE_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_single_graph_20260901/outputs}"
: "${RS_OUTPUT_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_20260921/outputs}"
: "${RS_ER_INPUT:=$RS_OUTPUT_ROOT/er_benchmark/pilot_N1000_seed21901001}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$RS_PROJECT_ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=${NSLOTS:-1}
export OPENBLAS_NUM_THREADS=${NSLOTS:-1}
export MKL_NUM_THREADS=${NSLOTS:-1}
export NUMEXPR_NUM_THREADS=${NSLOTS:-1}
export MPLCONFIGDIR=${HOFFMAN_SCRATCH}/random_schrodinger_20260921/.mplconfig
export XDG_CACHE_HOME=${HOFFMAN_SCRATCH}/random_schrodinger_20260921/.cache

case "$SGE_TASK_ID" in
  1)
    CASE=er
    INPUT_DIR="$RS_ER_INPUT"
    ;;
  2)
    CASE=pa
    INPUT_DIR="$RS_SOURCE_ROOT/affine_alpha_2p0"
    ;;
  3)
    CASE=cl
    INPUT_DIR=
    ;;
  *)
    echo "unsupported SGE_TASK_ID=$SGE_TASK_ID" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="$RS_OUTPUT_ROOT/tunneling_applications/$CASE"
mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
if [ -n "$INPUT_DIR" ]; then
  python3 tools/hoffman_random_schrodinger.py apply-tunneling \
    --case "$CASE" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --project-root "$RS_PROJECT_ROOT"
else
  python3 tools/hoffman_random_schrodinger.py apply-tunneling \
    --case "$CASE" \
    --output-dir "$OUTPUT_DIR" \
    --project-root "$RS_PROJECT_ROOT"
fi
