#!/bin/bash
set -euo pipefail
: "${HOFFMAN_SCRATCH:?set HOFFMAN_SCRATCH to your scheduler scratch root}"
: "${HOFFMAN_HOME:?set HOFFMAN_HOME to your scheduler home root}"
: "${LP_PROJECT_ROOT:=${HOFFMAN_SCRATCH}/small_world_mobility_edge_pa_mobility_20260823/project}"
: "${LP_RUN_ROOT:=${HOFFMAN_SCRATCH}/linear_pa_two_part_20260924_v3}"
: "${LP_CONFIG:=$LP_PROJECT_ROOT/experiments/linear_pa_two_part_20260924/config.json}"
cd "$LP_PROJECT_ROOT"
mkdir -p "$LP_RUN_ROOT/logs"
ALPHA_COUNT=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["alpha_grid"]))' "$LP_CONFIG")
LEDGER="$LP_RUN_ROOT/JOB_LEDGER.csv"
if [[ ! -e "$LEDGER" ]]; then
  echo 'submitted_utc,job_id,stage,dependency,configuration,output_root,status_at_submission' > "$LEDGER"
fi
record_job() {
  local job_id="$1" stage="$2" dependency="$3" configuration="$4" output_root="$5"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$job_id,$stage,$dependency,$configuration,$output_root,SUBMITTED" >> "$LEDGER"
}
FREEZE_JOB=$(qsub -terse tools/hoffman_linear_pa_two_part_freeze.sh)
FREEZE_BASE=${FREEZE_JOB%%.*}
record_job "$FREEZE_BASE" freeze '' "$LP_CONFIG" "$LP_RUN_ROOT/freeze/executed_source_v3"
TEST_JOB=$(qsub -terse -hold_jid "$FREEZE_BASE" tools/hoffman_linear_pa_two_part_tests.sh)
TEST_BASE=${TEST_JOB%%.*}
record_job "$TEST_BASE" validation "$FREEZE_BASE" "$LP_CONFIG" "$LP_RUN_ROOT/validation"
PILOT_JOB=$(qsub -terse -hold_jid "$TEST_BASE" tools/hoffman_linear_pa_two_part_pilot.sh)
PILOT_BASE=${PILOT_JOB%%.*}
record_job "$PILOT_BASE" end_to_end_pilot "$TEST_BASE" experiments/linear_pa_two_part_20260924/pilot_config.json "$LP_RUN_ROOT/pilot"
P1_500_JOB=$(qsub -terse -hold_jid "$PILOT_BASE" -t "1-$ALPHA_COUNT" -tc 2 tools/hoffman_linear_pa_two_part_part1_n500.sh)
P1_500_BASE=${P1_500_JOB%%.*}
record_job "$P1_500_BASE" part1_N500 "$PILOT_BASE" "$LP_CONFIG" "$LP_RUN_ROOT/cases/N_500"
P1_10K_JOB=$(qsub -terse -hold_jid "$PILOT_BASE" -t "1-$ALPHA_COUNT" -tc 1 tools/hoffman_linear_pa_two_part_part1_n10000.sh)
P1_10K_BASE=${P1_10K_JOB%%.*}
record_job "$P1_10K_BASE" part1_N10000 "$PILOT_BASE" "$LP_CONFIG" "$LP_RUN_ROOT/cases/N_10000"
P2_500_JOB=$(qsub -terse -hold_jid "$P1_500_BASE" -t "1-$ALPHA_COUNT" -tc 2 tools/hoffman_linear_pa_two_part_part2_n500.sh)
P2_500_BASE=${P2_500_JOB%%.*}
record_job "$P2_500_BASE" part2_N500 "$P1_500_BASE" "$LP_CONFIG" "$LP_RUN_ROOT/cases/N_500"
P2_10K_JOB=$(qsub -terse -hold_jid "$P1_10K_BASE" -t "1-$ALPHA_COUNT" -tc 1 tools/hoffman_linear_pa_two_part_part2_n10000.sh)
P2_10K_BASE=${P2_10K_JOB%%.*}
record_job "$P2_10K_BASE" part2_N10000 "$P1_10K_BASE" "$LP_CONFIG" "$LP_RUN_ROOT/cases/N_10000"
AGG_JOB=$(qsub -terse -hold_jid "$P2_500_BASE,$P2_10K_BASE" tools/hoffman_linear_pa_two_part_aggregate.sh)
AGG_BASE=${AGG_JOB%%.*}
record_job "$AGG_BASE" aggregate "$P2_500_BASE;$P2_10K_BASE" "$LP_CONFIG" "$LP_RUN_ROOT/report"
echo "freeze=$FREEZE_BASE tests=$TEST_BASE pilot=$PILOT_BASE part1_N500=$P1_500_BASE part1_N10000=$P1_10K_BASE part2_N500=$P2_500_BASE part2_N10000=$P2_10K_BASE aggregate=$AGG_BASE"
