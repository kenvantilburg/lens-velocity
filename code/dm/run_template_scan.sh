#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=python
#SBATCH --mail-type=END
#SBATCH --mail-user=cmondino@pitp.ca
#SBATCH --output=slurm_%j.out
##SBATCH --output=slurm_%A_%a.out


# set number of steps equal to number of cores
#n_steps=$SLURM_CPUS_ON_NODE
n_steps=40
echo "Number of steps = " $n_steps

#module load anaconda3

##To run do:
##sbatch run_template_scan.sh

#python template_scan.py 20 $n_steps 0 &
#python template_scan.py 20 $n_steps 1 &
#python template_scan.py 20 $n_steps 2 &
#python template_scan.py 20 $n_steps 3 &

max_step=39
for i_step in $(eval echo "{0..$max_step..1}")
    do
        echo "Starting template scan $i_step"
        python template_scan.py 90 $n_steps $i_step &
    done
# set 'wait' barrier so entire job does not get killed when last srun command does
wait 

echo "All jobs completed"
