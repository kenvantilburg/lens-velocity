#!/usr/bin/env python
# coding: utf-8

from time import time as tictoc
import sys
from my_units import * 
from angular_fn import *
from sky_patch_class import *
from template_fn import *
from sim_setup_fn import *
from sim_injection_fn import *
from data_cleaning_fn import *
from sim_analysis_fn import *


HomeDir = '../'
DataDir = HomeDir+'data/'
ListDir = HomeDir+'lists/'
ListTauDir = ListDir+'/data_tau/'
ListDataChiSq = HomeDir+'lists/data_chi_sq/'


# # Setup


### Define the sky patches used for the analysis
### Parameters taken from Gaia Early Data Release 3: Structure and properties of the Magellanic Clouds (see Table 4)
LMC_sky_patch = sky_patch(81.28, -69.78, 5*degree, 50*kpc, 'LMC_disc_5', np.array([1.871, 0.391]), pm_esc = 0.2)
SMC_sky_patch = sky_patch(12.80, -73.15, 4*degree, 60*kpc, 'SMC_disc_4', np.array([0.686, -1.237]), pm_esc = 0.2)

all_sky_patches = [LMC_sky_patch, SMC_sky_patch]
data_file_names = [sky_patch.data_file_name for sky_patch in all_sky_patches]


### Load list of beta_t values and convert to radians
beta_t_list = np.genfromtxt(ListDir+'beta_t_list.dat', delimiter='\n')/10000*degree


### List of files with the data analysis result
list_files = os.listdir(ListTauDir) 
[ind_ra, ind_dec, ind_tau_ra, ind_tau_dec, ind_n, ind_tau_mon, ind_tau_mon_n] = range(7)

param_space_points = np.loadtxt(sys.argv[1], dtype='str') ### As columns of M_l r_l f_l

# ## Compute the chi sq for the LMC and SMC

# In[36]:


columns_res = ['ra', 'dec', 'beta_t', 'min_chi_sq']

for p in param_space_points:
    M_l = math.pow(10, float(p[0])/10)*MSolar
    r_l = math.pow(10, (-3 + float(p[1])/10))*pc
    f_l = math.pow(10, (-3 + float(p[2])/10))    
    
    beta_t_opt = fn_beta_t_opt(M_l, r_l, f_l, all_sky_patches)
    beta_t_opt_list = fn_beta_t_opt_list(beta_t_opt, beta_t_list)
    beta_t_opt_deg_list = (np.round(10000*beta_t_opt_list/degree)).astype(int).astype(str)
    
    for i_sp, data_file_name in enumerate(data_file_names):
        len_dfn = len(data_file_name)
        print('Computing the chi_sq for', data_file_name, ', point', p)
        sys.stdout.flush()

        chi_sq_data = []

        for i_beta_t, beta_t_deg in enumerate(beta_t_opt_deg_list):
            beta_t = beta_t_opt_list[i_beta_t]
            print('Looking for files for beta t = '+beta_t_deg)        
            sys.stdout.flush()
            ### Select only files corresponding to the correct beta_t and data_file_name
            list_files_sel = [file for file in list_files if (file[:len_dfn]==data_file_name) & (file[len_dfn+6:len_dfn+7+len(beta_t_deg)]==beta_t_deg+'_')]  
#            list_files_sel = [file for file in list_files if (file[:len_dfn]==data_file_name) & (file[len_dfn+6:len_dfn+12+len(beta_t_deg)]==beta_t_deg+'_fine2')]   # use only results from the second fine scanning  
            
            print(len(list_files_sel), ' files found.')
            sys.stdout.flush()

            tau_values_data = []
            for file in list_files_sel:
                tau_values_file = np.load(ListTauDir+file)
                if len(tau_values_file) > 0:
                    tau_values_data.extend(tau_values_file)

            if len(tau_values_data)>0:
                chi_sq_i = fn_chi_sq(M_l, r_l, beta_t, np.array(tau_values_data), all_sky_patches[i_sp])                
                chi_sq_data.append(chi_sq_i[np.argmin(chi_sq_i[:, 3])])

        if len(chi_sq_data)>0:
            chi_sq_data = np.array(chi_sq_data); 
            res_df = pd.DataFrame(np.array([chi_sq_data[np.argmin(chi_sq_data[:, 3])]]), columns=columns_res)
            res_df.to_csv(ListDataChiSq+data_file_name+'_chisq_'+p[0]+'_'+p[1]+'_'+p[2]+'.csv', index=False)
    
    print('\n')
    sys.stdout.flush()

