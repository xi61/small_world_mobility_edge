#!/bin/bash
#$ -cwd
#$ -N sw_pa_m_summary
#$ -j y
#$ -o logs/m_sensitivity_summary_$JOB_ID.log
#$ -l h_rt=04:00:00
#$ -l h_data=8G
#$ -pe shared 1

set -euo pipefail

: "${USER:?USER must be set}"
HOFFMAN_SCRATCH="${HOFFMAN_SCRATCH:-/u/scratch/${USER:0:1}/${USER}}"
HOFFMAN_HOME="${HOFFMAN_HOME:-/u/home/${USER:0:1}/${USER}}"
REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
: "${PA_PROJECT_ROOT:?set PA_PROJECT_ROOT}"
: "${PA_OUTPUT_ROOT:?set PA_OUTPUT_ROOT}"
: "${PA_MODEL_FAMILY:?set PA_MODEL_FAMILY}"
: "${PA_PARAMETER_VALUE:?set PA_PARAMETER_VALUE}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ Hoffman module}"
cd "$PA_PROJECT_ROOT"
export PYTHONPATH=src
export MPLCONFIGDIR=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/.mplconfig_m_summary
export XDG_CACHE_HOME=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/.cache_m_summary
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python3 -m small_world_mobility_edge.preferential_attachment.mobility_edge_m_sensitivity_summary \
  --root "$PA_OUTPUT_ROOT" \
  --family "$PA_MODEL_FAMILY" \
  --n "${PA_N:-5000}" \
  --parameter-value "$PA_PARAMETER_VALUE" \
  --m-values 2 5 10 20
