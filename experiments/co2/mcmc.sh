#!/bin/bash parallel-gps
source ~/pycharm-2020.3.3/bin/activate parallel-gps

py_script=pssgp.experiments.co2.mcmc
step_size=0.01
noise_variance=0.05
mcmc=HMC
data_dir=~/PycharmProjects/parallel-gps/experiments/co2


for qp_order in 3; do
  python -m $py_script --step_size=$step_size --mcmc=$mcmc --model=GP --qp_order=$qp_order --noise_variance=$noise_variance --dtype=float64 --device="/gpu:1" --data_dir=$data_dir
  python -m $py_script --step_size=$step_size --mcmc=$mcmc --model=PSSGP --qp_order=$qp_order --noise_variance=$noise_variance --dtype=float64 --device="/gpu:1" --data_dir=$data_dir
  python -m $py_script --step_size=$step_size --mcmc=$mcmc --model=SSGP --qp_order=$qp_order --noise_variance=$noise_variance --dtype=float64 --device="/cpu:0" --data_dir=$data_dir
done