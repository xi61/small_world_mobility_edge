#!/bin/bash
#$ -cwd
#$ -N sw_pa_hubedge
#$ -j y
#$ -o logs/hub_edge_$JOB_ID.log
#$ -l h_rt=04:00:00
#$ -l h_data=12G
#$ -pe shared 1

set -euo pipefail

: "${USER:?USER must be set}"
HOFFMAN_SCRATCH="${HOFFMAN_SCRATCH:-/u/scratch/${USER:0:1}/${USER}}"
HOFFMAN_HOME="${HOFFMAN_HOME:-/u/home/${USER:0:1}/${USER}}"
REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
: "${PA_PROJECT_ROOT:?set PA_PROJECT_ROOT}"
: "${PA_OUTPUT_ROOT:?set PA_OUTPUT_ROOT}"
: "${PA_GRAPH_SEED_BASE:?set PA_GRAPH_SEED_BASE}"
: "${PA_ATTACHMENT_ZETA:?set PA_ATTACHMENT_ZETA}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ Hoffman module}"
cd "$PA_PROJECT_ROOT"
export PYTHONPATH=src
export MPLCONFIGDIR=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/.mplconfig_hub_edge
export XDG_CACHE_HOME=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/.cache_hub_edge
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python3 tools/hoffman_pa_nonlinear_hub_edge.py
