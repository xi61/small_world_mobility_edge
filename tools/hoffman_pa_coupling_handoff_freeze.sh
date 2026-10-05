#!/bin/bash
#$ -cwd
#$ -N pa_cp_freeze
#$ -j y
#$ -o logs/freeze_$JOB_ID.log
#$ -l h_rt=00:15:00
#$ -l h_data=2G
#$ -pe shared 1
set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${RS_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${RS_HANDOFF_ROOT:=${HOFFMAN_SCRATCH}/random_schrodinger_pa_coupling_and_anderson_20260924_v1}"
: "${RS_FREEZE_LABEL:=executed_clean_pa_source_v6}"
source /u/local/Modules/default/init/bash
module load "${PYTHON_MODULE:?set PYTHON_MODULE to a Python 3.11+ module}"
cd "$RS_PROJECT_ROOT"
FREEZE_DIR="$RS_HANDOFF_ROOT/freeze/$RS_FREEZE_LABEL"
mkdir -p "$FREEZE_DIR" "$RS_HANDOFF_ROOT/logs"
find src tools tests experiments -type f \( -name '*.py' -o -name '*.sh' -o -name '*.json' \) ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$FREEZE_DIR/source_configuration_checksums.sha256"
PYTHONPATH=src python3 -c 'from pathlib import Path; from small_world_mobility_edge.random_schrodinger.common import source_configuration_identity; print(source_configuration_identity(Path.cwd())["sha256"])' > "$FREEZE_DIR/runtime_source_configuration_id.txt"
SOURCE_CONFIGURATION_ID=$(tr -d '[:space:]' < "$FREEZE_DIR/runtime_source_configuration_id.txt")
SOURCE_CONFIGURATION_CHECKSUM_ID=$(sha256sum "$FREEZE_DIR/source_configuration_checksums.sha256" | awk '{print $1}')
BUNDLE_PATHS=(README.md pyproject.toml src tools tests experiments)
[[ -e PARALLEL_WORK_STATUS.md ]] && BUNDLE_PATHS+=(PARALLEL_WORK_STATUS.md)
tar -czf "$FREEZE_DIR/source_configuration_bundle.tar.gz" --exclude='*/__pycache__' --exclude='*.pyc' "${BUNDLE_PATHS[@]}"
sha256sum "$FREEZE_DIR/source_configuration_bundle.tar.gz" > "$FREEZE_DIR/source_configuration_bundle.sha256"
{
  echo "captured_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "job_id=${JOB_ID:-}"
  echo "python=$(python3 --version 2>&1)"
  echo "uname=$(uname -a)"
  echo "source_configuration_id=$SOURCE_CONFIGURATION_ID"
  echo "source_configuration_checksum_manifest_id=$SOURCE_CONFIGURATION_CHECKSUM_ID"
  echo "project_root=$RS_PROJECT_ROOT"
  echo "scope=clean_pa_only; diagonal_disorder_and_rrg_paused"
} > "$FREEZE_DIR/environment_summary.txt"
module list > "$FREEZE_DIR/module_list.txt" 2>&1 || true
python3 -m pip freeze --all > "$FREEZE_DIR/python_packages.txt"
lscpu > "$FREEZE_DIR/lscpu.txt" 2>&1 || true
env | LC_ALL=C sort > "$FREEZE_DIR/sge_environment.txt"
ENVIRONMENT_ID=$(sha256sum "$FREEZE_DIR/environment_summary.txt" "$FREEZE_DIR/module_list.txt" "$FREEZE_DIR/python_packages.txt" "$FREEZE_DIR/lscpu.txt" | sha256sum | awk '{print $1}')
python3 -c 'import hashlib,json,pathlib,sys; d=pathlib.Path(sys.argv[1]); payload={"status":"COMPLETE","freeze_label":sys.argv[2],"source_configuration_id":sys.argv[3],"environment_id":sys.argv[4],"source_configuration_bundle_sha256":hashlib.sha256((d/"source_configuration_bundle.tar.gz").read_bytes()).hexdigest(),"scope":"clean PA only; disorder/RRG paused","historical_source_reconstruction_claimed":False}; (d/"freeze_manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")' "$FREEZE_DIR" "$RS_FREEZE_LABEL" "$SOURCE_CONFIGURATION_ID" "$ENVIRONMENT_ID"
sha256sum "$FREEZE_DIR"/* > "$FREEZE_DIR/freeze_artifact_checksums.sha256"
echo "SOURCE_CONFIGURATION_ID=$SOURCE_CONFIGURATION_ID"
echo "ENVIRONMENT_ID=$ENVIRONMENT_ID"
