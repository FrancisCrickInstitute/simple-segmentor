#!/bin/bash
# in: config filepath
config_file=$1

job_name=`basename $config_file .yml`
output_path="experiments/$job_name/output/$job_name.job"
error_path="experiments/$job_name/output/$job_name.job"
mkdir -p "experiments/$job_name/"
mkdir -p "experiments/$job_name/output/"

sbatchcmd="sbatch --job-name=$job_name --output=$output_path --error=$error_path scripts/job_run_experiment.sh $config_file"
echo
echo "SBATCH SUBMISSION COMMAND: $sbatchcmd"
echo
eval $sbatchcmd

