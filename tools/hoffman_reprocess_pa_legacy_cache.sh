#!/bin/bash
#$ -cwd
#$ -N sw_pa_legacy
#$ -j y
#$ -o logs/legacy_reprocess_$JOB_ID.log
#$ -l h_rt=12:00:00
#$ -l h_data=16G
#$ -pe shared 1

set -euo pipefail

: "${USER:?USER must be set}"
HOFFMAN_SCRATCH="${HOFFMAN_SCRATCH:-/u/scratch/${USER:0:1}/${USER}}"
HOFFMAN_HOME="${HOFFMAN_HOME:-/u/home/${USER:0:1}/${USER}}"
REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ Hoffman module}"

PROJECT=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project
SOURCE=${HOFFMAN_HOME}/small_world_mobility_edge_cache/pa_full_20260813
PROVENANCE=${HOFFMAN_HOME}/small_world_mobility_edge_cache/pa_full_20260813_provenance/durable_copy
OUTPUT=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/outputs/preferential_attachment/mobility_edge

cd "$PROJECT"
export PYTHONPATH=src
export MPLCONFIGDIR=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/.mplconfig_legacy
export XDG_CACHE_HOME=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/.cache_legacy
export OMP_NUM_THREADS=${NSLOTS:-2}
export OPENBLAS_NUM_THREADS=${NSLOTS:-2}
export MKL_NUM_THREADS=${NSLOTS:-2}
export NUMEXPR_NUM_THREADS=${NSLOTS:-2}
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

python3 tools/hoffman_pa_cached_reprocess.py \
  --source-root "$SOURCE" \
  --provenance-root "$PROVENANCE" \
  --output-root "$OUTPUT"
