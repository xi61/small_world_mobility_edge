#!/bin/bash
#$ -cwd
#$ -N sw_pa_me_agg
#$ -j y
#$ -o logs/aggregate_$JOB_ID.log
#$ -l h_rt=12:00:00
#$ -l h_data=16G
#$ -pe shared 1

set -euo pipefail

: "${USER:?USER must be set}"
HOFFMAN_SCRATCH="${HOFFMAN_SCRATCH:-/u/scratch/${USER:0:1}/${USER}}"
HOFFMAN_HOME="${HOFFMAN_HOME:-/u/home/${USER:0:1}/${USER}}"
REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
: "${PA_PROJECT_ROOT:?set PA_PROJECT_ROOT}"
: "${PA_OUTPUT_ROOT:?set PA_OUTPUT_ROOT}"
: "${PA_MODEL_FAMILY:?set PA_MODEL_FAMILY}"
: "${PA_GRAPH_SEED_BASE:?set PA_GRAPH_SEED_BASE}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ Hoffman module}"
: "${PA_PYTHON:=python3}"

cd "$PA_PROJECT_ROOT"
mkdir -p ${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/logs
export PYTHONPATH=src
export MPLCONFIGDIR=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/.mplconfig_aggregate
export XDG_CACHE_HOME=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/.cache_aggregate
export OMP_NUM_THREADS=${NSLOTS:-2}
export OPENBLAS_NUM_THREADS=${NSLOTS:-2}
export MKL_NUM_THREADS=${NSLOTS:-2}
export NUMEXPR_NUM_THREADS=${NSLOTS:-2}
export PA_CHUNK_SIZE=${PA_CHUNK_SIZE:-5}
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
"$PA_PYTHON" tools/hoffman_pa_mobility_edge_aggregate.py
