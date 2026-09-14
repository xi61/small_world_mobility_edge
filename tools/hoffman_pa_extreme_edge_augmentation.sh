#!/bin/bash
#$ -cwd
#$ -N sw_pa_edgeaug
#$ -j y
#$ -o logs/extreme_edge_augmentation_$JOB_ID.$TASK_ID.log
#$ -l h_rt=24:00:00
#$ -l h_data=8G
#$ -pe shared 4

set -euo pipefail
: "${PA_PROJECT_ROOT:?set PA_PROJECT_ROOT}"
: "${PA_OUTPUT_ROOT:?set PA_OUTPUT_ROOT}"
: "${PA_EXTREME_EDGE_OUTPUT_ROOT:?set PA_EXTREME_EDGE_OUTPUT_ROOT}"
: "${PA_MODEL_FAMILY:?set PA_MODEL_FAMILY}"
: "${PA_PARAMETER_NAME:?set PA_PARAMETER_NAME}"
: "${PA_PARAMETER_VALUE:?set PA_PARAMETER_VALUE}"
: "${PA_TARGET_GRAPH_COUNT:?set PA_TARGET_GRAPH_COUNT}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ Hoffman module}"
cd "$PA_PROJECT_ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=${NSLOTS:-4}
export OPENBLAS_NUM_THREADS=${NSLOTS:-4}
export MKL_NUM_THREADS=${NSLOTS:-4}
export NUMEXPR_NUM_THREADS=${NSLOTS:-4}
python3 tools/hoffman_pa_extreme_edge_augmentation.py
