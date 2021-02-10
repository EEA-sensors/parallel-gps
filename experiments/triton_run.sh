#!/bin/bash

#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1

cd $WRKDIR/parallel-gps

module load cuda
module load cudnn
module load anaconda3/latest

# Create venv
mkdir venv
path_base_python=$(which python)
python -m venv $WRKDIR/parallel-gps/venv
source venv/bin/activate
pip install --upgrade pip

# Setup
python setup.py install

# Run
cd experiments
python triton_test.py