#!/bin/bash

#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1

cd $WRKDIR/parallel-gps

module load cuda
#module load cudnn
# module load anaconda3/latest
# module load python/3.7.4
module load anaconda/2020-05-tf2

# conda create -c nvidia -c defaults python==3.7 'cudatoolkit==11.0.*' 'cudnn>=8' -p /path-to-env

# Create venv
#mkdir venv
#path_base_python=$(which python)
#python -m venv $WRKDIR/parallel-gps/venv
#source venv/bin/activate
#pip install --upgrade pip
#
##
#pip install numpy
#pip install scipy
#pip install numba
#pip install tensorflow==2.4.1
#pip install gpflow

conda activate $WRKDIR/zz-env

# Setup
# python setup.py install
#python setup.py develop

# Run
cd experiments
python triton_test.py