#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:0
#SBATCH --mem=192G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

config_file=$1
pyscript="/camp/stp/ddt/working/howesj/simple-segmentor/experiment.py"
pypath="PYTHONPATH=/camp/stp/ddt/working/howesj/simple-segmentor"

# activate conda environment
source /camp/apps/eb/software/Anaconda3/2020.07/bin/activate /camp/stp/ddt/working/howesj/jpyenv

pycmd="PYTHONUNBUFFERED=1 $pypath python $pyscript --config $config_file"

echo "RUNNING PYTHON COMMAND: $pycmd"
echo

eval $pycmd

