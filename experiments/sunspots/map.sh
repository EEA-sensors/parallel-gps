#!/bin/bash parallel-gps
source ~/pycharm-2020.3.3/bin/activate parallel-gps

py_script=pssgp.experiments.sunspot.map
noise_variance=350.
data_dir=~/PycharmProjects/parallel-gps/experiments/
python -m $py_script --model=PSSGP --noise_variance=$noise_variance --dtype=float64 --device="/gpu:1" --data_dir=$data_dir
#sleep 10s
python -m $py_script --model=SSGP --noise_variance=$noise_variance --dtype=float64 --device="/cpu:0" --data_dir=$data_dir
#sleep 10s
python -m $py_script --model=GP --noise_variance=$noise_variance --dtype=float64 --device="/gpu:1" --data_dir=$data_dir
