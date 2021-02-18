#!/bin/bash parallel-gps
source ~/pycharm-2020.3.3/bin/activate parallel-gps

py_script=pssgp.experiments.toy_models.mcmc
step_size=0.01
n_runs=10

for mcmc in HMC MALA NUTS; do
  for cov in Matern32 Matern52 RBF; do
    python -m $py_script --step_size=$step_size --n_runs=$n_runs --mcmc=$mcmc --model=PSSGP --cov=$cov --rbf_order=6 --rbf_balance_iter=10 --qp_order=6 --data_model=SINE --noise_variance=0.1 --dtype=float32 --device="/gpu:0"
    sleep 10s
    python -m $py_script --step_size=$step_size --n_runs=$n_runs --mcmc=$mcmc --model=SSGP --cov=$cov --rbf_order=6 --rbf_balance_iter=10 --qp_order=6 --data_model=SINE --noise_variance=0.1 --dtype=float32 --device="/cpu:0"
    sleep 10s
    python -m $py_script --step_size=$step_size --n_runs=$n_runs --mcmc=$mcmc --model=GP --cov=$cov --rbf_order=6 --rbf_balance_iter=10 --qp_order=6 --data_model=SINE --noise_variance=0.1 --dtype=float64 --device="/gpu:0"
    sleep 10s
  done
done
