#!/bin/bash

#SBATCH -c 4
#SBATCH --gres=v100:1
#SBATCH --time=01:00:00

cd $WRKDIR/

module load cuda
module load anaconda-latest
