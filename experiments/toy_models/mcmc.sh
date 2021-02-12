#!/bin/bash parallel-gps
source ~/pycharm-2020.3.3/bin/activate parallel-gps

py_script=pssgp.experiments.toy_models.mcmc
step_size=0.01
cov=Matern32

for mcmc in HMC MALA NUTS; do
  for cov in RBF, Matern32; do
    python -m $py_script --step_size=$step_size --mcmc=$mcmc --model=PSSGP --cov=$cov --rbf_order=6 --rbf_balance_iter=10 --qp_order=6 --data_model=SINE --noise_variance=0.1 --dtype=float32 --device="/gpu:0" &
    python -m $py_script --step_size=$step_size --mcmc=$mcmc --model=SSGP --cov=$cov --rbf_order=6 --rbf_balance_iter=10 --qp_order=6 --data_model=SINE --noise_variance=0.1 --dtype=float32 --device="/gpu:1"

    python -m $py_script --step_size=$step_size --mcmc=$mcmc --model=GP --cov=$cov --rbf_order=6 --rbf_balance_iter=10 --qp_order=6 --data_model=SINE --noise_variance=0.1 --dtype=float32 --device="/gpu:0"
  done
done
