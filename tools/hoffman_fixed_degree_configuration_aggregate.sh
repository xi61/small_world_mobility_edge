#!/bin/bash
#$ -cwd
#$ -N sw_cfgagg
#$ -j y
#$ -l h_rt=04:00:00
#$ -l h_data=16G
#$ -pe shared 2

set -euo pipefail
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ Hoffman module}"

: "${PROJECT_ROOT:?set PROJECT_ROOT to the Hoffman project copy}"
: "${PARTIAL_DIR:?set PARTIAL_DIR for per-pair NPZ files}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for aggregate outputs}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR" logs .mplconfig .cache
export PYTHONPATH=src
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export OMP_NUM_THREADS="${NSLOTS:-2}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-2}"
export MKL_NUM_THREADS="${NSLOTS:-2}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-2}"

python3 -m small_world_mobility_edge.fixed_degree_configuration_joint_tail \
  --n "${N:-5000}" \
  --tau "${TAU:-1.5}" \
  --degree "${DEGREE:-20}" \
  --num-graphs "${NUM_GRAPHS:-500}" \
  --swaps-per-edge "${SWAPS_PER_EDGE:-10}" \
  --max-tries-factor "${MAX_TRIES_FACTOR:-20}" \
  --tau-convention "${TAU_CONVENTION:-density_exponent_ccdf_tau_minus_one}" \
  --cutoff-coefficient "${CUTOFF_COEFFICIENT:-1.0}" \
  --aggregate-partials \
  --partial-dir "$PARTIAL_DIR" \
  --output-dir "$OUTPUT_DIR"
