#!/bin/bash
#$ -cwd
#$ -N pa_mf_aggregate
#$ -j y
#$ -l h_rt=02:00:00
#$ -l h_data=8G
#$ -pe shared 1
source "${PA_MF_PROJECT_ROOT:?}/tools/hoffman_pa_multifractality_env.sh"
"$PA_MF_PYTHON" tools/hoffman_pa_multifractality.py --config "$PA_MF_CONFIG" aggregate --output-root "$PA_MF_RUN_ROOT/cases" --report-dir "$PA_MF_RUN_ROOT/report"
