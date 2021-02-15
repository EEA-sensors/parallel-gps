#!/bin/bash

#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:v100:1

cd $WRKDIR/parallel-gps/experiments/toy_models

if [ ! -d "./results" ]
then
    echo "results history folder not exists, will mkdir"
    mkdir ./results
fi

module load cuda
module load cudnn
module load anaconda/2020-05-tf2

CONDA_BASE=$(conda info --base) ; source $CONDA_BASE/etc/profile.d/conda.sh

conda activate $WRKDIR/zz-env

py_script=pssgp.experiments.toy_models.mcmc

step_size=$1
n_runs=$2
mcmc=$3
cov=$4
device=$5

echo "Running GP_${mcmc}_${cov}"

python -m $py_script --step_size=$step_size --n_runs=$n_runs --mcmc=$mcmc --model=GP --cov=$cov --rbf_order=6 --rbf_balance_iter=10 --qp_order=6 --data_model=SINE --noise_variance=0.1 --dtype=float32 --device=$device
