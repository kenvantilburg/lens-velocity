#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=03:00:00
#SBATCH --job-name=python
#SBATCH --output=slurm_%j.out


module purge
module load python/intel/3.8.6

n_iter=39

for i_iter in $(eval echo "{0..$n_iter..1}")
    do
        python pairs_and_accels.py $i_iter &
    done
wait


echo "All jobs completed"
