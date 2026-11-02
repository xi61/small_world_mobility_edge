#!/bin/bash
# Sourced by the SGE jobs. Use a pre-installed Python >=3.11 environment.
set -euo pipefail
: "${PA_MF_PROJECT_ROOT:?Missing project root}"
: "${PA_MF_RUN_ROOT:?Missing run root}"
: "${PA_MF_CONFIG:?Missing config}"
: "${PA_MF_PYTHON:?Missing Python executable}"
cd "$PA_MF_PROJECT_ROOT"
export PYTHONPATH="$PA_MF_PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$PA_MF_RUN_ROOT/matplotlib/${JOB_ID:-local}.${SGE_TASK_ID:-0}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR"
