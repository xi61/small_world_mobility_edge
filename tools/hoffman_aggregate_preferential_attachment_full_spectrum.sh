#!/bin/bash
#$ -cwd
#$ -N sw_pa_agg
#$ -j y
#$ -o logs/pa_aggregate.log
#$ -l h_rt=08:00:00
#$ -l h_data=32G
#$ -pe shared 2

set -euo pipefail

: "${USER:?USER must be set}"
HOFFMAN_SCRATCH="${HOFFMAN_SCRATCH:-/u/scratch/${USER:0:1}/${USER}}"
HOFFMAN_HOME="${HOFFMAN_HOME:-/u/home/${USER:0:1}/${USER}}"
REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ Hoffman module}"
cd ${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_full_20260813
mkdir -p logs
export PYTHONPATH=src
export MPLCONFIGDIR=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_full_20260813/.mplconfig_aggregate
export XDG_CACHE_HOME=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_full_20260813/.cache_aggregate
export OMP_NUM_THREADS=${NSLOTS:-2}
export OPENBLAS_NUM_THREADS=${NSLOTS:-2}
export MKL_NUM_THREADS=${NSLOTS:-2}
export NUMEXPR_NUM_THREADS=${NSLOTS:-2}
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python3 tools/hoffman_preferential_attachment_full_spectrum_aggregate.py
