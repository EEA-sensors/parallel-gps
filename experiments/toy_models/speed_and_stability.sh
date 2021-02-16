#!/bin/bash parallel-gps
source ~/pycharm-2020.3.3/bin/activate parallel-gps

mesh_size=10
py_script=pssgp.experiments.toy_models.speed_and_stability

for cov in Matern32 Matern52 RBF; do
  python -m $py_script --model=SSGP --cov=$cov --rbf_order=6 --rbf_balance_iter=10 --qp_order=6 --data_model=SINE --noise_variance=0.1 --dtype=float64 --device="/cpu:0" --mesh_size=$mesh_size
  python -m $py_script --model=PSSGP --cov=$cov --rbf_order=6 --rbf_balance_iter=10 --qp_order=6 --data_model=SINE --noise_variance=0.1 --dtype=float64 --device="/gpu:0" --mesh_size=$mesh_size
  python -m $py_script --model=GP --cov=$cov --rbf_order=6 --rbf_balance_iter=10 --qp_order=6 --data_model=SINE --noise_variance=0.1 --dtype=float64 --device="/gpu:1" --mesh_size=$mesh_size
  sleep 10s
done
