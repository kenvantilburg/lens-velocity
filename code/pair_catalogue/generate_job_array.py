import numpy as np
import sys

run_steps = np.arange(0,3386,30)
run_steps = np.append(run_steps, 3386)

count = 1 

for run_i in range(len(run_steps)-1):
    run_init = run_steps[run_i]
    run_final = run_steps[run_i+1]
    script_lines = ['#!/bin/bash',
                '#SBATCH --nodes=1',
                            '#SBATCH --ntasks-per-node=40',
                            '#SBATCH --time=00:12:00',
                            '#SBATCH --job-name=python', 
                            '#SBATCH --output=slurm_%j.out', 
                            '#SBATCH --mem=179GB',
                            '\n',
                            'module purge;',
                            'module load anaconda3/2020.07;',
                            '\n',
                            'for i_step in $(eval echo "{'+str(run_init)+'..'+str(run_final)+'..1}")',
                            '    do',
                            '        python pairs_and_accels.py $i_step &',
                            '    done',
                            'wait', '\n',
                            'echo "All jobs completed"']
    script_file = open('run_cat_'+str(count)+'.sh', "w")
    for line in script_lines:
        script_file.write(line + "\n")
    script_file.close()
    count += 1