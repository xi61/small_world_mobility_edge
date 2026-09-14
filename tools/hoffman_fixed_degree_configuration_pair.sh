#!/bin/bash
#$ -cwd
#$ -N sw_cfgpair
#$ -j y
#$ -l h_rt=24:00:00
#$ -l h_data=12G
#$ -pe shared 4

set -euo pipefail
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ Hoffman module}"

: "${PROJECT_ROOT:?set PROJECT_ROOT to the Hoffman project copy}"
: "${PARTIAL_DIR:?set PARTIAL_DIR for per-pair NPZ files}"

cd "$PROJECT_ROOT"
mkdir -p "$PARTIAL_DIR" logs .mplconfig .cache
export PYTHONPATH=src
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export OMP_NUM_THREADS="${NSLOTS:-4}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-4}"
export MKL_NUM_THREADS="${NSLOTS:-4}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-4}"

python3 -m small_world_mobility_edge.fixed_degree_configuration_joint_tail \
  --n "${N:-5000}" \
  --tau "${TAU:-1.5}" \
  --degree "${DEGREE:-20}" \
  --num-graphs "${NUM_GRAPHS:-500}" \
  --swaps-per-edge "${SWAPS_PER_EDGE:-10}" \
  --max-tries-factor "${MAX_TRIES_FACTOR:-20}" \
  --tau-convention "${TAU_CONVENTION:-density_exponent_ccdf_tau_minus_one}" \
  --cutoff-coefficient "${CUTOFF_COEFFICIENT:-1.0}" \
  --pair-index "$((SGE_TASK_ID - 1))" \
  --partial-dir "$PARTIAL_DIR"
