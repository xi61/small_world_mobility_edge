#!/bin/bash
#$ -cwd
#$ -N sw_e1_agg
#$ -j y
#$ -o logs/e1_projector_aggregate.log
#$ -l h_rt=00:30:00
#$ -l h_data=4G
#$ -pe shared 1

set -euo pipefail

: "${USER:?USER must be set}"
HOFFMAN_SCRATCH="${HOFFMAN_SCRATCH:-/u/scratch/${USER:0:1}/${USER}}"
HOFFMAN_HOME="${HOFFMAN_HOME:-/u/home/${USER:0:1}/${USER}}"
REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ Hoffman module}"
cd ${HOFFMAN_SCRATCH}/small_world_mobility_edge_evec_qscan_20260812
export PYTHONPATH=src
export MPLCONFIGDIR=${HOFFMAN_SCRATCH}/small_world_mobility_edge_evec_qscan_20260812/.mplconfig_e1_projector
export XDG_CACHE_HOME=${HOFFMAN_SCRATCH}/small_world_mobility_edge_evec_qscan_20260812/.cache_e1_projector
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" logs
python3 tools/hoffman_chung_lu_e1_projector_aggregate.py
