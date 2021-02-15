#!/bin/bash

# Run this script after the environment has been setup by `triton_setup.sh`

#SBATCH -o logs/mcmc_ssgp.log

cd $WRKDIR/parallel-gps/experiments/toy_models

if [ ! -d "./results" ]
then
    echo "results history folder not exists, will mkdir"
    mkdir ./results
fi

module load anaconda/2020-05-tf2

conda activate $WRKDIR/zz-env

py_script=pssgp.experiments.toy_models.mcmc

step_size=$1
n_runs=$2
mcmc=$3
cov=$4
device=$5

python -m $py_script --step_size=$step_size --n_runs=$n_runs --mcmc=$mcmc --model=SSGP --cov=$cov --rbf_order=6 --rbf_balance_iter=10 --qp_order=6 --data_model=SINE --noise_variance=0.1 --dtype=float32 --device=$device
