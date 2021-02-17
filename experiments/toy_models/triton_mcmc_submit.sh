#!/bin/bash
cd ~/PycharmProjects/parallel-gps/experiments/toy_models

bash ~/PycharmProjects/parallel-gps/venv/bin/activate

py_script=pssgp.experiments.toy_models.mcmc
step_size=0.01
n_runs=10

for mcmc in HMC; do
  for cov in Matern32; do
#    python -m $py_script --step_size=$step_size --n_runs=$n_runs --mcmc=$mcmc --model=PSSGP --cov=$cov --rbf_order=6 --rbf_balance_iter=10 --qp_order=6 --data_model=SINE --noise_variance=0.1 --dtype=float32 --device="/gpu:0"
#    CUDA_VISIBLE_DEVICES="" python -m $py_script --step_size=$step_size --n_runs=$n_runs --mcmc=$mcmc --model=SSGP --cov=$cov --rbf_order=6 --rbf_balance_iter=10 --qp_order=6 --data_model=SINE --noise_variance=0.1 --dtype=float32 --device="/cpu:0"
    python -m $py_script --step_size=$step_size --n_runs=$n_runs --mcmc=$mcmc --model=GP --cov=$cov --rbf_order=6 --rbf_balance_iter=10 --qp_order=6 --data_model=SINE --noise_variance=0.1 --dtype=float64 --device="/gpu:0"
  done
done
