import numpy as np
import sys

### Generate scripts to run simulations on the cluster for the points in the file given by sys.argv[1]
### We run 100 simulations for the LMC and 100 simulations for the SMC
### per parameter space point, divided in groups of 25 to be run on one node (even though the node has 40 cores, it seems better to run less simulations per node)

param_space_points = np.loadtxt(sys.argv[1], dtype='str') ### As columns of M_l r_l f_l
data_file_names = [['LMC', '02:00:00'], ['SMC', '01:30:00']] 
sim_steps = np.arange(1, 175, 25)

count = 1

for p in param_space_points:
    print(p)
    for data_file in data_file_names:
        for sim_i in range(len(sim_steps)-1):
            sim_init = sim_steps[sim_i]; sim_final = sim_steps[sim_i+1]-1;

            script_lines = ['#!/bin/bash',
                            '#SBATCH --nodes=1',
                            '#SBATCH --time='+data_file[1],
                            '#SBATCH --job-name=python', 
                            '#SBATCH --output=slurm_%j.out', '\n',
                            'for i_step in $(eval echo "{'+str(sim_init)+'..'+str(sim_final)+'..1}")',
                            '    do',
                            '        python simulation.py '+data_file[0]+' '+p[0]+' '+p[1]+' '+p[2]+' $i_step &',
                            '    done',
                            'wait', '\n',
                            'echo "All jobs completed"']

            script_file = open('run_sim_'+str(count)+'.sh', "w")

            for line in script_lines:
                script_file.write(line + "\n")
            script_file.close()

            count += 1
