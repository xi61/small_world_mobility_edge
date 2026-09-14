#!/bin/bash
#$ -cwd
#$ -N sw_evec_q2
#$ -j y
#$ -o logs/evec_qscan_$TASK_ID.log
#$ -l h_rt=24:00:00
#$ -l h_data=8G
#$ -pe shared 4
#$ -t 1-300

set -euo pipefail

: "${USER:?USER must be set}"
HOFFMAN_SCRATCH="${HOFFMAN_SCRATCH:-/u/scratch/${USER:0:1}/${USER}}"
HOFFMAN_HOME="${HOFFMAN_HOME:-/u/home/${USER:0:1}/${USER}}"
REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ Hoffman module}"
cd ${HOFFMAN_SCRATCH}/small_world_mobility_edge_evec_qscan_20260812
mkdir -p logs
export PYTHONPATH=src
export MPLCONFIGDIR=${HOFFMAN_SCRATCH}/small_world_mobility_edge_evec_qscan_20260812/.mplconfig
export XDG_CACHE_HOME=${HOFFMAN_SCRATCH}/small_world_mobility_edge_evec_qscan_20260812/.cache
export OMP_NUM_THREADS=${NSLOTS:-4}
export OPENBLAS_NUM_THREADS=${NSLOTS:-4}
export MKL_NUM_THREADS=${NSLOTS:-4}
export VECLIB_MAXIMUM_THREADS=${NSLOTS:-4}
export NUMEXPR_NUM_THREADS=${NSLOTS:-4}
export CHUNK_SIZE=5
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python3 tools/hoffman_chung_lu_eigenvector_percentile_chunk.py
