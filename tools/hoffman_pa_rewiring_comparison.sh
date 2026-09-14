#!/bin/bash
#$ -cwd
#$ -N sw_pa_rewire_compare
#$ -j y
#$ -o logs/rewiring_comparison_$JOB_ID.log
#$ -l h_rt=06:00:00
#$ -l h_data=16G
#$ -pe shared 1

set -euo pipefail

: "${USER:?USER must be set}"
HOFFMAN_SCRATCH="${HOFFMAN_SCRATCH:-/u/scratch/${USER:0:1}/${USER}}"
HOFFMAN_HOME="${HOFFMAN_HOME:-/u/home/${USER:0:1}/${USER}}"
REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
: "${PA_PROJECT_ROOT:?set PA_PROJECT_ROOT}"
: "${PA_PARAMETER_DIR:?set PA_PARAMETER_DIR}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ Hoffman module}"
cd "$PA_PROJECT_ROOT"
export PYTHONPATH=src
export MPLCONFIGDIR=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/.mplconfig_rewire_compare
export XDG_CACHE_HOME=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/.cache_rewire_compare
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python3 -m small_world_mobility_edge.preferential_attachment.mobility_edge_rewiring_comparison \
  --parameter-dir "$PA_PARAMETER_DIR"
