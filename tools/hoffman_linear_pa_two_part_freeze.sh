#!/bin/bash
#$ -cwd
#$ -N lpa2v3_freeze
#$ -j y
#$ -o logs/freeze_$JOB_ID.log
#$ -l h_rt=00:20:00
#$ -l h_data=2G
#$ -pe shared 1
set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${HOFFMAN_HOME:?set HOFFMAN_HOME to your scheduler home root}"
: "${LP_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${LP_RUN_ROOT:=${HOFFMAN_SCRATCH}/linear_pa_two_part_20260924_v3}"
: "${LP_FREEZE_LABEL:=executed_source_v3}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$LP_PROJECT_ROOT"
FREEZE_DIR="$LP_RUN_ROOT/freeze/$LP_FREEZE_LABEL"
mkdir -p "$FREEZE_DIR" "$LP_RUN_ROOT/logs"
find src tools tests experiments -type f \( -name '*.py' -o -name '*.sh' -o -name '*.json' \) ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$FREEZE_DIR/source_configuration_checksums.sha256"
PYTHONPATH=src python3 -c 'from pathlib import Path; from small_world_mobility_edge.random_schrodinger.common import source_configuration_identity; print(source_configuration_identity(Path.cwd())["sha256"])' > "$FREEZE_DIR/runtime_source_configuration_id.txt"
SOURCE_CONFIGURATION_ID=$(tr -d '[:space:]' < "$FREEZE_DIR/runtime_source_configuration_id.txt")
tar -czf "$FREEZE_DIR/source_configuration_bundle.tar.gz" --exclude='*/__pycache__' --exclude='*.pyc' README.md pyproject.toml src tools tests experiments PARALLEL_WORK_STATUS.md
sha256sum "$FREEZE_DIR/source_configuration_bundle.tar.gz" > "$FREEZE_DIR/source_configuration_bundle.sha256"
{
  echo "captured_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "job_id=${JOB_ID:-}"
  echo "python=$(python3 --version 2>&1)"
  echo "source_configuration_id=$SOURCE_CONFIGURATION_ID"
  echo "project_root=$LP_PROJECT_ROOT"
  echo "scope=clean_affine_linear_PA_N500_N10000"
} > "$FREEZE_DIR/environment_summary.txt"
module list > "$FREEZE_DIR/module_list.txt" 2>&1 || true
python3 -m pip freeze --all > "$FREEZE_DIR/python_packages.txt"
lscpu > "$FREEZE_DIR/lscpu.txt" 2>&1 || true
python3 -c 'import hashlib,json,pathlib,sys; d=pathlib.Path(sys.argv[1]); payload={"status":"COMPLETE","freeze_label":sys.argv[2],"source_configuration_id":sys.argv[3],"source_configuration_bundle_sha256":hashlib.sha256((d/"source_configuration_bundle.tar.gz").read_bytes()).hexdigest(),"scope":"clean affine linear PA two-part experiment","historical_outputs_modified":False}; (d/"freeze_manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")' "$FREEZE_DIR" "$LP_FREEZE_LABEL" "$SOURCE_CONFIGURATION_ID"
sha256sum "$FREEZE_DIR"/* > "$FREEZE_DIR/freeze_artifact_checksums.sha256"
echo "SOURCE_CONFIGURATION_ID=$SOURCE_CONFIGURATION_ID"
