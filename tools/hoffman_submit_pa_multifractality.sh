#!/bin/bash
set -euo pipefail
: "${PA_MF_PROJECT_ROOT:?Set the absolute repository checkout path}"
: "${PA_MF_SOURCE_ROOT:?Set the absolute cached clean-PA cases directory}"
: "${PA_MF_RUN_ROOT:?Set a new absolute output directory}"
: "${PA_MF_CONFIG:=$PA_MF_PROJECT_ROOT/experiments/pa_multifractality_20261021/config.json}"
: "${PA_MF_PYTHON:=$(command -v python3)}"
export PA_MF_PROJECT_ROOT PA_MF_SOURCE_ROOT PA_MF_RUN_ROOT PA_MF_CONFIG PA_MF_PYTHON
mkdir -p "$PA_MF_RUN_ROOT/logs"
cd "$PA_MF_PROJECT_ROOT"
count="$("$PA_MF_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["alpha_grid"]))' "$PA_MF_CONFIG")"
environment="PA_MF_PROJECT_ROOT,PA_MF_SOURCE_ROOT,PA_MF_RUN_ROOT,PA_MF_CONFIG,PA_MF_PYTHON"
validation_job="$(qsub -terse -v "$environment" -o "$PA_MF_RUN_ROOT/logs" tools/hoffman_pa_multifractality_validate.sh)"
case_job="$(qsub -terse -v "$environment" -o "$PA_MF_RUN_ROOT/logs" -hold_jid "$validation_job" -t "1-$count" tools/hoffman_pa_multifractality_case.sh)"
aggregate_job="$(qsub -terse -v "$environment" -o "$PA_MF_RUN_ROOT/logs" -hold_jid "${case_job%%.*}" tools/hoffman_pa_multifractality_aggregate.sh)"
printf 'validation_job=%s\ncase_array_job=%s\naggregate_job=%s\n' "$validation_job" "$case_job" "$aggregate_job"
