#!/usr/bin/env python
# coding: utf-8

# In[1]:


from time import time as tictoc
import sys
from sky_patch_class import *
from my_units import * 
from angular_fn import *
from template_fn import *
from sim_setup_fn import *
from sim_injection_fn import *
from data_cleaning_fn import *
from sim_analysis_fn import *

HomeDir = '../'
DataDir = HomeDir+'data/'
ListDir = HomeDir+'lists/'
ListResDir = HomeDir+'lists/sim/'


# Notebook to run the simulation for a given paramter space point (M_l, r_l, f_l) on a given stellar target (LMC or SMC). This notebook should be converted to a python script and run on a cluster. Some parameters for the simulation are read in from the command line. For example, the python script to do the 0-th simulation on the LMC for (M_l, r_l, f_l) = (10^8 M_solar, 1 pc, 1) should be run as:
# 
# python simulation.py LMC 80 30 30 0
# 
# Takes as input:
# * data_file_name+'_clean.npy', the cleaned data generated in the notebook data_cleaning.ipynb
# 
# where data_file_name are 'LMC_disc_5' or 'SMC_disc_4'. Generate as output the minumum chi^2 (optimal test statistic) with its corresponding beta_t value and location, i.e.
# * data_file_name+'_80_30_30_0.csv'

# # Automatic simulation evaluation

# ## Preamble

# In[3]:


tic = tictoc()

### Define the sky patches used for the analysis
### Parameters taken from the paper Gaia Early Data Release 3: Structure and properties of the Magellanic Clouds (see Table 4)
LMC_sky_patch = sky_patch(81.28, -69.78, 5*degree, 50*kpc, 'LMC_disc_5', np.array([1.871, 0.391]), pm_esc = 0.2, sigma_pm = 0.125)
SMC_sky_patch = sky_patch(12.80, -73.15, 4*degree, 60*kpc, 'SMC_disc_4', np.array([0.686, -1.237]), pm_esc = 0.2, sigma_pm = 0.105)

all_sky_patches = [LMC_sky_patch, SMC_sky_patch]


# In[ ]:


### Read in paramters from the command line
sky_patch_name = sys.argv[1] # LMC or SMC
M_l = math.pow(10, float(sys.argv[2])/10)*MSolar
r_l = math.pow(10, (-3 + float(sys.argv[3])/10))*pc
f_l = math.pow(10, (-3 + float(sys.argv[4])/10))


# In[5]:


### Define sky patch to use in this simulation
if sky_patch_name == 'LMC':    
    sky_p = LMC_sky_patch
    print(' ********** Running the '+sys.argv[5]+'th simulation analysis on the LMC **********\n')
elif sky_patch_name == 'SMC':
    sky_p = SMC_sky_patch
    print(' ********** Running the '+sys.argv[5]+'th simulation analysis on the SMC **********\n')
else:
    print('ERROR: wrong name provided for the sky patch!')
sys.stdout.flush()


# In[ ]:


result_file_name = sky_p.data_file_name+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+sys.argv[4]+'_'+sys.argv[5]
print('Result will be saved in the file '+result_file_name)


# In[8]:


### Parameters for data cleaning
beta_kernel_sub_0 = 0.1*degree; beta_kernel_sub = 0.06*degree;   # gaussian kernels for background subtraction 
n_sigma_out = 3;                                                 # number of sigmas for outlier removal
n_iter_sub = 3;                                                  # number of iterations for the background subtraction
disc_radius_no_edge = sky_p.disc_radius - beta_kernel_sub_0 - 2*(n_iter_sub+1)*beta_kernel_sub
gmag_bin_size=0.1; rad_bin_size=1                                # g mag and radial bins used to compute the effective dispersion

### Parameters for the template scan and simulation
n_betat = 6; n_betat_inj = 6; 
min_beta_t = 0.006*degree                                        # for point-like lenses, inject signal on stars around the lens within n_betat*min_beta_t 
min_mask = 0.01*degree                                           # minimum radius of the mask used to compute the template 
beta_step = 1/9;                                                 # beta_t step used in the second step of the fine scanning, 
                                                                 # needed to determine positions around the lens locations where to compute the template

n_lens_max = 200                                                 # maximum number of the lenses to keep for the signal injection and analysis (the closest to the observer are kept)
gmag_bin_size_noise = 0.05                                       # g mag bin size used to inject the noise 


# In[9]:


### Load list of beta_t values and convert to radians
beta_t_list = np.genfromtxt(ListDir+'beta_t_list.dat', delimiter='\n')/10000*degree  #np.load(ListDir+'beta_t_list.npy')/10000*degree 
### Optimal value of beta_t for the lens population parameters
beta_t_opt = fn_beta_t_opt(M_l, r_l, f_l, all_sky_patches)
### Find the beta_t values from the beta_t_list closest to the optimal beta_t
beta_t_opt_list = fn_beta_t_opt_list(beta_t_opt, beta_t_list)

print('\nOptimal beta_t value:', str(beta_t_opt/degree)[:7], ' deg.') 
print('Computing the template for beta_t =', str(str(beta_t_opt_list/degree)))
sys.stdout.flush()


# In[ ]:


### Loading the data -- loading an npy file is much faster than loading the csv file with pd.rad_csv
data_np = np.load(DataDir+sky_p.data_file_name+'_clean.npy')
columns_df = ['ra', 'dec', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'phot_g_mean_mag', 'pmra_sub', 'pmdec_sub']
data = pd.DataFrame(data_np, columns=columns_df)


# ## Execution

# In[9]:


### Generating the lens population
n_lens, lens_pop = fn_lens_population(M_l, f_l, sky_p, n_lens_max)
print(n_lens, 'lenses in front of the stellar target.'); sys.stdout.flush()    


# In[ ]:


columns_res = ['ra', 'dec', 'beta_t', 'min_chi_sq']

if n_lens == 0: ### If there are no lenses in front of the stellar target, skip the analysis and set the resulting chi^2 to zero
    res_df = pd.DataFrame([np.zeros(len(columns_res))], columns=columns_res)
    res_df.to_csv(ListResDir+result_file_name+'.csv', index=False)
    toc = tictoc()    
    print('Simulation done in', str(toc - tic), 's.')        
    sys.stdout.flush()          
else:
    print('\nPreparing the mock data..'); sys.stdout.flush()    
    ### Injecting the noise
    fn_noise_inj(data, sky_p, gmag_bin_size_noise, rad_bin_size, noise=True)
    ### Injecting the noise
    fn_signal_inj(data, M_l, r_l, n_lens, lens_pop, sky_p, n_betat_inj, min_beta_t)    

    # For SMC only: cut on the pm to remove stars from the foreground globular clusters
    if sky_p.data_file_name == 'SMC_disc_4':   
        orig_len = len(data)
        data = data[(data['pmra_sim'] < 0.685 + 2) & (data['pmra_sim'] > 0.685 - 2) &
                    (data['pmdec_sim'] < -1.230 + 2) & (data['pmdec_sim'] > -1.230 - 2)]
    
    ### Prepare the mock data for the iterative background subtraction and outlier removal
    disc_pix, nb_pixel_list, n = fn_prepare_back_sub(data, sky_p.disc_center, sky_p.disc_radius, beta_kernel_sub)
    
    print('\nBackground subtraction..'); sys.stdout.flush()    
    ### Iterative background subtraction and outlier removal
    for i in range(n_iter_sub):        
        fn_back_field_sub(data, disc_pix, nb_pixel_list, n, beta_kernel=beta_kernel_sub, sub=False, sim=True) ### sub=True can be used only after this function has been already called once with sub=False
        data, f_out = fn_rem_outliers(data, sky_p.pm_esc, sky_p.distance/kpc, n_sigma_out, sim=True)
        print('Iter '+str(i)+' -- fraction of outliers removed: '+str(f_out*100)[:7]+' %')
    fn_back_field_sub(data, disc_pix, nb_pixel_list, n, beta_kernel=beta_kernel_sub, sub=False, sim=True)

    ### Remove stars at the boundary to avoid edge effect due to gaussian kernel field subtraction
    data = fn_rem_edges(data, sky_p.disc_center, disc_radius_no_edge)       
    ### Compute the effective weights
    fn_effective_w(data, sky_p.disc_center, gmag_bin_size, rad_bin_size, sim=True)
    
    data_final = np.array([data['ra'].to_numpy(), data['dec'].to_numpy(), (data['pm_eff_error'].to_numpy())**2, 
                           data['pmra_sim'].to_numpy()/data['pm_eff_error'].to_numpy()**2, data['pmdec_sim'].to_numpy()/data['pm_eff_error'].to_numpy()**2]).T
    
    ### Run the analysis on the mock data to compute the template at the lens locations and compute the chi^2
    chi_sq = [];
    for beta_t in beta_t_opt_list:
        chi_sq.extend(fn_run_analysis(data_final, beta_t, n_betat, min_mask, beta_step, M_l, r_l, n_lens, lens_pop, sky_p)); 
    chi_sq = np.array(chi_sq)
    
    ### Save the result into a file
    res_df = pd.DataFrame([chi_sq[np.argmin(chi_sq[:, 3])]], columns=columns_res)
    res_df.to_csv(ListResDir+result_file_name+'.csv', index=False)

    toc = tictoc()    
    print('Simulation done in', str(toc - tic), 's.')        
    sys.stdout.flush()

