#!/bin/bash parallel-gps

step_size=0.01
n_runs=10

for mcmc in HMC MALA NUTS; do
  for cov in Matern32 Matern52 RBF; do
    sbatch triton_mcmc_PSSGP $step_size $n_runs $mcmc $cov "/gpu:0"
    sbatch triton_mcmc_SSGP $step_size $n_runs $mcmc $cov "/cpu:0"
    sbatch triton_mcmc_GP $step_size $n_runs $mcmc $cov "/gpu:0"
  done
done