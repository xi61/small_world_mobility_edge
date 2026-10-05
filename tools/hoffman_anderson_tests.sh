#!/bin/bash
#$ -cwd
#$ -N and_tests
#$ -j y
#$ -o logs/tests_$JOB_ID.log
#$ -l h_rt=01:00:00
#$ -l h_data=6G
#$ -pe shared 1
set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${RS_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${RS_RUN_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_repair_and_extension_20260924_v1}"
: "${RS_FREEZE_MANIFEST:=$RS_RUN_ROOT/freezes/executed_anderson_source_v1/freeze_manifest.json}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$RS_PROJECT_ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$RS_RUN_ROOT/validation" "$RS_RUN_ROOT/logs"
python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("status")=="COMPLETE"' "$RS_FREEZE_MANIFEST"
python3 -m unittest -v tests.test_anderson_repair_extension
python3 -c 'import json,os,pathlib,platform,sys; from small_world_mobility_edge.random_schrodinger.common import source_configuration_identity,utc_now; root=pathlib.Path(sys.argv[1]); freeze=json.load(open(sys.argv[2])); sid=source_configuration_identity(root)["sha256"]; assert sid==freeze["source_configuration_id"]; out=pathlib.Path(sys.argv[3]); out.write_text(json.dumps({"status":"COMPLETE","validation_passed":True,"test_count":10,"job_id":os.environ.get("JOB_ID"),"source_configuration_id":sid,"python_version":platform.python_version(),"finished_at_utc":utc_now()},indent=2,sort_keys=True)+"\n")' "$RS_PROJECT_ROOT" "$RS_FREEZE_MANIFEST" "$RS_RUN_ROOT/validation/run_manifest.json"
