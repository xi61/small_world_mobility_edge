#!/bin/bash
#$ -cwd
#$ -N sw_pa_band_agg
#$ -j y
#$ -o logs/abnormal_band_aggregate_$JOB_ID.log
#$ -l h_rt=02:00:00
#$ -l h_data=6G
#$ -pe shared 1

set -euo pipefail

: "${USER:?USER must be set}"
HOFFMAN_SCRATCH="${HOFFMAN_SCRATCH:-/u/scratch/${USER:0:1}/${USER}}"
HOFFMAN_HOME="${HOFFMAN_HOME:-/u/home/${USER:0:1}/${USER}}"
REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$PWD}}"
: "${PA_PROJECT_ROOT:?set PA_PROJECT_ROOT}"
: "${PA_MICROSCOPE_OUTPUT_ROOT:?set PA_MICROSCOPE_OUTPUT_ROOT}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ Hoffman module}"
cd "$PA_PROJECT_ROOT"
export PYTHONPATH=src
export MPLCONFIGDIR=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/.mplconfig_abnormal_band_aggregate
export XDG_CACHE_HOME=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/.cache_abnormal_band_aggregate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python3 tools/hoffman_pa_abnormal_band_microscope_aggregate.py   --output-root "$PA_MICROSCOPE_OUTPUT_ROOT"
python3 tools/validate_pa_abnormal_band_microscope.py   --output-root "$PA_MICROSCOPE_OUTPUT_ROOT"
