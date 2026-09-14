#!/bin/bash
#$ -cwd
#$ -N sw_e1_proj
#$ -j y
#$ -o logs/e1_projector_$TASK_ID.log
# The one-graph pilot used 0.75 GB and 12 seconds on four cores.  A single
# 2-GB slot makes these short tasks easier to dispatch while retaining ample
# memory and runtime margins.
#$ -l h_rt=00:15:00
#$ -l h_data=2G
#$ -pe shared 1
#$ -t 1-250
#$ -tc 25

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
export MPLCONFIGDIR=${HOFFMAN_SCRATCH}/small_world_mobility_edge_evec_qscan_20260812/.mplconfig_e1_projector
export XDG_CACHE_HOME=${HOFFMAN_SCRATCH}/small_world_mobility_edge_evec_qscan_20260812/.cache_e1_projector
export OMP_NUM_THREADS=${NSLOTS:-4}
export OPENBLAS_NUM_THREADS=${NSLOTS:-4}
export MKL_NUM_THREADS=${NSLOTS:-4}
export VECLIB_MAXIMUM_THREADS=${NSLOTS:-4}
export NUMEXPR_NUM_THREADS=${NSLOTS:-4}
export CHUNK_SIZE=2
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python3 tools/hoffman_chung_lu_e1_projector_chunk.py
