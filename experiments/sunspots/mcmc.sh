#!/bin/bash parallel-gps
source ~/pycharm-2020.3.3/bin/activate parallel-gps

py_script=pssgp.experiments.sunspot.mcmc
step_size=1.
noise_variance=300.
data_dir=~/PycharmProjects/parallel-gps/experiments/
mcmc=HMC

python -m $py_script --step_size=$step_size --mcmc=$mcmc --model=PSSGP --noise_variance=$noise_variance --dtype=float64 --device="/gpu:0" --data_dir=$data_dir
#sleep 10s
#python -m $py_script --step_size=$step_size --mcmc=$mcmc --model=SSGP --noise_variance=$noise_variance --dtype=float32 --device="/cpu:0" --data_dir=$data_dir
#sleep 10s
#python -m $py_script --step_size=$step_size --mcmc=$mcmc --model=GP --noise_variance=$noise_variance --dtype=float64 --device="/gpu:0" --data_dir=$data_dir
sleep 10s
