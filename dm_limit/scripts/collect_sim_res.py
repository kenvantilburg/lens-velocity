#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import os
import sys

HomeDir = '../'
ListResDir = HomeDir+'lists/sim/'




param_space_points = np.loadtxt(sys.argv[1], dtype='str') ### As columns of M_l r_l f_l
data_file_names = ['LMC_disc_5', 'SMC_disc_4'] 
max_n_sim = 200



columns_res = ['ra', 'dec', 'beta_t', 'min_chi_sq']

for p in param_space_points:
    for data_file_name in data_file_names:

        chisq_list = []       
        for i in range(max_n_sim):
            file_name = ListResDir+data_file_name+'_'+p[0]+'_'+p[1]+'_'+p[2]+'_'+str(i)+'.csv'
            if os.path.isfile(file_name):
                df = pd.read_csv(file_name) 
                chisq_list.append(df.to_numpy()[0])
        if len(chisq_list) > 0:  
            print('Saving collective result for', data_file_name, ', point', p)
            ### Check if a file with all the simulations already exist. If yes, append the new simulations result to the same file
            all_file_name = ListResDir+data_file_name+'_'+p[0]+'_'+p[1]+'_'+p[2]+'_all.csv'
            if os.path.isfile(all_file_name):
                print('File', all_file_name, 'already exists. Appending..')
                res_df = pd.read_csv(all_file_name)                
                res_df.append(pd.DataFrame(np.array(chisq_list), columns=columns_res)).to_csv(all_file_name, index=False)
            ### Otherwise write a new file
            else:
                print(np.array(chisq_list).shape)
                print('File', all_file_name, 'does not exists. Creating new file..')
                res_df = pd.DataFrame(np.array(chisq_list), columns=columns_res)
                res_df.to_csv(all_file_name, index=False)
            ### Remove files for the individual simulations
            for i in range(max_n_sim):
                file_name = ListResDir+data_file_name+'_'+p[0]+'_'+p[1]+'_'+p[2]+'_'+str(i)+'.csv'
                if os.path.isfile(file_name):
                    os.remove(file_name)






