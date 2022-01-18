#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --job-name=python
#SBATCH --output=slurm_%j.out


module purge
module load python/intel/3.8.6

python pairs_and_accels.py 0
