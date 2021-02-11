#!/bin/bash

# Run this script after the environment has been setup by `triton_setup.sh`

#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --gres=gpu:v100:1

# plot shoud never be flagged in Triton

cd $WRKDIR/parallel-gps/experiments

if [ ! -d "./results" ]
then
    echo "results history folder not exists, will mkdir"
    mkdir ./results
fi

module load cuda
module load cudnn
module load anaconda/2020-05-tf2

conda activate $WRKDIR/zz-env

python sinusoidal_regression.py --Nm $1 --Np $2 --model $3 --cov $4 --inference_method $5 --n_samples $6 --burnin $7 --rbf_order $8 --rbf_balance_iter $9