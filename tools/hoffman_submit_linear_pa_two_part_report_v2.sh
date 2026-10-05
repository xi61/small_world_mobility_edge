#!/bin/bash
set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${HOFFMAN_HOME:?set HOFFMAN_HOME to your scheduler home root}"
: "${LP_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${LP_REPORT_RUN_ROOT:=${HOFFMAN_SCRATCH}/linear_pa_two_part_20260924_v3_reporting_v5}"
cd "$LP_PROJECT_ROOT"
mkdir -p "$LP_REPORT_RUN_ROOT/logs"
LEDGER="$LP_REPORT_RUN_ROOT/JOB_LEDGER.csv"
if [[ ! -e "$LEDGER" ]]; then
  echo 'submitted_utc,job_id,stage,dependency,configuration,output_root,status_at_submission' > "$LEDGER"
fi
record_job() {
  local job_id="$1" stage="$2" dependency="$3" configuration="$4" output_root="$5"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$job_id,$stage,$dependency,$configuration,$output_root,SUBMITTED" >> "$LEDGER"
}
FREEZE_JOB=$(qsub -terse \
  -v LP_RUN_ROOT="$LP_REPORT_RUN_ROOT",LP_FREEZE_LABEL=executed_reporting_source_v5 \
  tools/hoffman_linear_pa_two_part_freeze.sh)
FREEZE_BASE=${FREEZE_JOB%%.*}
record_job "$FREEZE_BASE" reporting_freeze '' 'report-only revision' "$LP_REPORT_RUN_ROOT/freeze/executed_reporting_source_v5"
TEST_JOB=$(qsub -terse -hold_jid "$FREEZE_BASE" \
  -v LP_RUN_ROOT="$LP_REPORT_RUN_ROOT",LP_FREEZE_MANIFEST="$LP_REPORT_RUN_ROOT/freeze/executed_reporting_source_v5/freeze_manifest.json" \
  tools/hoffman_linear_pa_two_part_tests.sh)
TEST_BASE=${TEST_JOB%%.*}
record_job "$TEST_BASE" reporting_validation "$FREEZE_BASE" 'report-only revision' "$LP_REPORT_RUN_ROOT/validation"
REPORT_JOB=$(qsub -terse -hold_jid "$TEST_BASE" tools/hoffman_linear_pa_two_part_report_v2.sh)
REPORT_BASE=${REPORT_JOB%%.*}
record_job "$REPORT_BASE" aggregate_report_v2 "$TEST_BASE" 'v3 scientific outputs; v5 reporting source' "$LP_REPORT_RUN_ROOT/report"
echo "freeze=$FREEZE_BASE tests=$TEST_BASE report=$REPORT_BASE"
