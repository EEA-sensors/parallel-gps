#!/bin/bash

# Run this script to give the results shown in paper
# Example: bash triton_submit.sh "sinu" or bash triton_submit.sh "co2"

#flags.DEFINE_integer('Nm', 400, 'Number of measurements.')
#flags.DEFINE_integer('Np', 500, 'Number of predictions.')
#flags.DEFINE_string('model', ModelEnum.SSGP.value, 'Select model to run. Options are gp, ssgp, and pssgp.')
#flags.DEFINE_string('cov', CovFuncEnum.Matern32.value, 'Covariance function.')
#flags.DEFINE_string('inference_method', InferenceMethodEnum.MAP.value, 'How to learn hyperparameters. MAP or HMC.')
#flags.DEFINE_integer('n_samples', 100, 'Number of HMC samples')
#flags.DEFINE_integer('burnin', 100, 'Burning-in steps of HMC')
#flags.DEFINE_integer('rbf_order', 6, 'Order of ss-RBF approximation.', lower_bound=1)
#flags.DEFINE_integer('rbf_balance_iter', 10, 'Iterations of RBF balancing.', lower_bound=1)

experiment=$1
Nm=$2
Np=$3
time=$4

if [ $experiment == "sinu" ]
then
    sbatch triton_run_sinu.sh $Nm $Np "GP" "Matern32" "MAP" 100 100 6 10 --time=$time
    sbatch triton_run_sinu.sh $Nm $Np "GP" "Matern32" "HMC" 100 100 6 10 --time=$time
    sbatch triton_run_sinu.sh $Nm $Np "GP" "RBF" "MAP" 100 100 6 10 --time=$time
    sbatch triton_run_sinu.sh $Nm $Np "GP" "RBF" "HMC" 100 100 6 10 --time=$time

    sbatch triton_run_sinu.sh $Nm $Np "SSGP" "Matern32" "MAP" 100 100 6 10 --time=$time
    sbatch triton_run_sinu.sh $Nm $Np "SSGP" "Matern32" "HMC" 100 100 6 10 --time=$time
    sbatch triton_run_sinu.sh $Nm $Np "SSGP" "RBF" "MAP" 100 100 6 10 --time=$time
    sbatch triton_run_sinu.sh $Nm $Np "SSGP" "RBF" "HMC" 100 100 6 10 --time=$time

    sbatch triton_run_sinu.sh $Nm $Np "PSSGP" "Matern32" "MAP" 100 100 6 10 --time=$time
    sbatch triton_run_sinu.sh $Nm $Np "PSSGP" "Matern32" "HMC" 100 100 6 10 --time=$time
    sbatch triton_run_sinu.sh $Nm $Np "PSSGP" "RBF" "MAP" 100 100 6 10 --time=$time
    sbatch triton_run_sinu.sh $Nm $Np "PSSGP" "RBF" "HMC" 100 100 6 10 --time=$time
elif [ $1 == "co2" ]
then
    echo "soon..."
else
    echo "not defined experiment."
    exit 1
fi
