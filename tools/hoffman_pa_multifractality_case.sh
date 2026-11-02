#!/bin/bash
#$ -cwd
#$ -N pa_mf_case
#$ -j y
#$ -l h_rt=04:00:00
#$ -l h_data=16G
#$ -pe shared 1
source "${PA_MF_PROJECT_ROOT:?}/tools/hoffman_pa_multifractality_env.sh"
: "${PA_MF_SOURCE_ROOT:?Missing cached cases directory}"
"$PA_MF_PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["validation_passed"]' "$PA_MF_RUN_ROOT/validation/validation_manifest.json"
alpha="$("$PA_MF_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["alpha_grid"][int(sys.argv[2])-1])' "$PA_MF_CONFIG" "${SGE_TASK_ID:?}")"
"$PA_MF_PYTHON" tools/hoffman_pa_multifractality.py --config "$PA_MF_CONFIG" case --alpha "$alpha" --source-root "$PA_MF_SOURCE_ROOT" --output-root "$PA_MF_RUN_ROOT/cases"
