#!/bin/bash
#$ -cwd
#$ -N pa_mf_validate
#$ -j y
#$ -l h_rt=00:20:00
#$ -l h_data=4G
#$ -pe shared 1
source "${PA_MF_PROJECT_ROOT:?}/tools/hoffman_pa_multifractality_env.sh"
"$PA_MF_PYTHON" -m unittest discover -s tests -p 'test_pa_multifractality*.py'
"$PA_MF_PYTHON" tools/hoffman_pa_multifractality.py --config "$PA_MF_CONFIG" validate --output-dir "$PA_MF_RUN_ROOT/validation"
