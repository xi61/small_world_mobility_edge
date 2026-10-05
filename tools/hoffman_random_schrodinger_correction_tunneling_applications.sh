#!/bin/bash
#$ -cwd
#$ -N rs_corr_tapp
#$ -j y
#$ -o logs/tunneling_application_$JOB_ID.$TASK_ID.log
#$ -l h_rt=01:00:00
#$ -l h_data=6G
#$ -pe shared 1
#$ -t 1-4
#$ -tc 2

set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${RS_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${RS_SOURCE_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_single_graph_20260901/outputs}"
: "${RS_OLD_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_20260921}"
: "${RS_CORRECTION_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_correction_20260922_v1}"
: "${RS_FREEZE_MANIFEST:=$RS_CORRECTION_ROOT/freeze/executed_correction_source/freeze_manifest.json}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$RS_PROJECT_ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$RS_CORRECTION_ROOT/logs"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("status")=="COMPLETE" and d.get("validation_passed") is True' "$RS_CORRECTION_ROOT/tunneling_validation/run_manifest.json"
case "${SGE_TASK_ID:-}" in
  1)
    python3 tools/hoffman_random_schrodinger_corrections.py --project-root "$RS_PROJECT_ROOT" --freeze-manifest "$RS_FREEZE_MANIFEST" tunneling-application --case er --input-dir "$RS_OLD_ROOT/outputs/er_benchmark/pilot_N1000_seed21901001" --output-dir "$RS_CORRECTION_ROOT/tunneling_applications/er"
    ;;
  2)
    python3 tools/hoffman_random_schrodinger_corrections.py --project-root "$RS_PROJECT_ROOT" --freeze-manifest "$RS_FREEZE_MANIFEST" tunneling-application --case pa --input-dir "$RS_SOURCE_ROOT/affine_alpha_2p0" --output-dir "$RS_CORRECTION_ROOT/tunneling_applications/pa"
    ;;
  3)
    python3 tools/hoffman_random_schrodinger_corrections.py --project-root "$RS_PROJECT_ROOT" --freeze-manifest "$RS_FREEZE_MANIFEST" tunneling-application --case cl --input-dir "$RS_OLD_ROOT/outputs/tunneling_applications/cl" --output-dir "$RS_CORRECTION_ROOT/tunneling_applications/cl"
    ;;
  4)
    python3 tools/hoffman_random_schrodinger_corrections.py --project-root "$RS_PROJECT_ROOT" --freeze-manifest "$RS_FREEZE_MANIFEST" paper-er --input-dir "$RS_OLD_ROOT/outputs/er_benchmark/pilot_N1000_seed21901001" --old-application-dir "$RS_OLD_ROOT/outputs/tunneling_applications/er" --output-dir "$RS_CORRECTION_ROOT/paper_er_nonempty"
    ;;
  *) echo "invalid SGE_TASK_ID=${SGE_TASK_ID:-unset}" >&2; exit 2 ;;
esac
