#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.coordinates import *
from astropy.time import Time
import astropy.units as u
import healpy as hp
from scipy.special import erf
from scipy import stats
import scipy as sp
from os import listdir
import gzip
import sys

import myUnitsCopy1 as myU # customized library for units. All dimensional variables are in GeV and GeV=1
#import MyUnits as myU # customized library for units. All dimensional variables are in GeV and GeV=1


# In[2]:


#edr3_data = './edr3_data'
#dr2_data = './dr2_data'
#hist_res_dir = './hist_stats/'
edr3_data = '/Users/crimondino/Dropbox (PI)/MyLensVelocity2/data/acc_catalog/edr3/'
dr2_data = '/Users/crimondino/Dropbox (PI)/MyLensVelocity2/data/acc_catalog/dr2/'
hist_res_dir = '/Users/crimondino/Dropbox (PI)/MyLensVelocity2/lists/hist_stats/'


# In[3]:


current_index=0


# In[5]:


### Read in the eDR3 file index from the command line
current_index = int(sys.argv[1]) # current index in list of edr3 files
print('\nReading in eDR3 file '+str(current_index)+'.'); sys.stdout.flush()


# # Loading in the Files

# ### Load a Single EDR3 File

# In[6]:


list_dr3_files = listdir(edr3_data)


# In[7]:


healpix_edr3_start = np.empty((len(list_dr3_files)),dtype= int)
healpix_edr3_end = np.empty((len(list_dr3_files)), dtype = int)

for i,file in enumerate(list_dr3_files):
    int_1 = int(file[11:17])
    int_2 = int(file[18:24])
    healpix_edr3_start[i] = int_1
    healpix_edr3_end[i] = int_2
    


# In[8]:


def get_source_ids(file_names):
    #given a list of EDR3 filenames, return the start and end source IDs corresponding to healpix level 12
    N_8 = 2**(59-16)
    
    start = np.array([x*N_8 for x in healpix_edr3_start], dtype = 'int')
    end = np.array([x*N_8 for x in healpix_edr3_end], dtype = 'int')
    return start, end


# In[9]:


def load_dr3_file(idx):
    return pd.read_csv(edr3_data + '/' + list_dr3_files[idx], compression = 'gzip')


# In[10]:


start, end = get_source_ids(list_dr3_files)


# ### Load Corresponding DR2 files

# In[11]:


list_dr2_files = np.array([file for file in listdir(dr2_data) if file[-7:]=='.csv.gz']) #select only files ending with 'csv.gz'


# In[12]:


def load_dr2_files(idx):
    #separate into two arrays of start/end source IDs
    strings = np.array([file.split('_') for file in list_dr2_files])
    sid_dr2_start = np.array([int(name) for name in strings[:,1]])
    sid_dr2_end = np.array([int(name[:-7]) for name in strings[:,2]])

    pass1 = np.where(start[idx] < sid_dr2_end)[0]
    pass2 = np.where(end[idx] < sid_dr2_start)[0]

    file_indices = np.setdiff1d(pass1, pass2)

    files_to_open = list_dr2_files[file_indices]
    print(str(len(files_to_open))+ ' corresponding files')
    return pd.concat((pd.read_csv(dr2_data+ '/' + str(f), compression = 'gzip') for f in files_to_open))


# ### Call the functions

# In[13]:


dr3 = load_dr3_file(current_index)
dr2 = load_dr2_files(current_index)


# # Generate Pair Catalogue

# For each star, we first locate accidental pairs by on-sky proximity. This is the condition
# $$|\theta_i - \theta_j| < \theta_\text{min}$$
# where $i$ is the index of the foreground star, and $j$ is a background star for a given foreground $i$.
# After this first cut, we then impose that the background candidate be behind the foreground at $n_\sigma$. 
# 
# $$\varpi_i - \varpi_j > n_\sigma \sqrt{\sigma_{\varpi_i}^2 + \sigma_{\varpi_j}^2}.$$
# 
# When $n_\sigma =2$, this corresponds to a 95% confidence level. We can tighten or relax these cuts in order to control the size/purity of the resulting pair catalogue.
# 
# The above assumes that $\sigma_{\varpi_i}$ and $\sigma_{\varpi_j}$ have zero correlation. A stricter condition would be to assume that they had correlation = 1. If so, then the above formula becomes 
# $$\varpi_i - n_\sigma \sigma_{\varpi_i} > \varpi_i + n_\sigma \sigma_{\varpi_j}.$$
# This results in fewer pairs.

# In[14]:


def generate_pair_cat(df, angle_cutoff, n_sigma):
    #Note that angle_cutoff is measured in arcseconds.
    
    ra_arr = np.asarray(df['ra'])
    dec_arr = np.asarray(df['dec'])
    coord1 = SkyCoord(ra_arr, dec_arr, unit = u.degree)
    
    #Search df for on-sky neighbors within angle_cutoff arsec
    z = search_around_sky(coord1, coord1, angle_cutoff*u.arcsec, storekdtree = False)
    
    #The above snippet will count a foreground star as its own neighbor, so we must remove them:
    idx = z[0][z[0] != z[1]]
    dub = z[1][z[0] != z[1]]
    
    df_fore = df.iloc[idx]
    df_back = df.iloc[dub]
    
    df_fore.reset_index(inplace = True, drop=True)
    df_back.reset_index(inplace = True, drop=True)
    
    #Define a function to iterate over the foreground/background df's and check if they satisfy the parallax condition

    is_behind = lambda par1, par2, err1, err2 : par1-par2 > n_sigma*np.sqrt(err1**2 + err2**2)
    is_behind_list = is_behind(df_fore['parallax'], df_back['parallax'], df_fore['parallax_error'], df_back['parallax_error'])
    
    #Keep pairs that satisfy the parallax condition within n_sigma. 
    df_fore = df_fore[is_behind_list]
    df_back = df_back[is_behind_list]
    
    #Concatenate the foreground and background list into one catalogue.
    new_fg_cols = [x+"_fg" for x in df_fore.columns]
    df_fore.columns= new_fg_cols
    
    new_bg_cols = [x+"_bg" for x in df_back.columns]
    df_back.columns= new_bg_cols
    
    pair_cat = pd.concat([df_fore,df_back], axis = 1)
    pair_cat.reset_index(inplace =True, drop = True)
    return pair_cat


# # Generate Acceleration Catalogue
# 

# In[15]:


def generate_pairs_list(dr3, dr2):
    ra_arr1 = np.asarray(dr3['ra'])
    dec_arr1 = np.asarray(dr3['dec'])

    ra_arr2 = np.asarray(dr2['ra'])
    dec_arr2 = np.asarray(dr2['dec'])
    
    coord1 = SkyCoord(ra_arr1, dec_arr1, unit = u.degree)
    coord2 = SkyCoord(ra_arr2, dec_arr2, unit = u.degree)
    
    z = search_around_sky(coord1, coord2, 3*u.arcsec, storekdtree = False)
    
    df1 = dr3.iloc[z[0]]
    df2 = dr2.iloc[z[1]]
    
    df1.reset_index(inplace = True, drop=True)
    df2.reset_index(inplace = True, drop=True)
    
    new_cols = [x+".1" for x in df2.columns]
    df2.columns= new_cols
    result = pd.concat([df1,df2], axis = 1)
    result = result[(result['astrometric_params_solved']>= 27) & (result['astrometric_params_solved.1']>= 27)]
    return result


# In[16]:


def propagate_back_linear(ra_g3, dec_g3, pmra_g3, pmdec_g3):
    """Takes EDR3 position and proper motion, and linearly propagates it by 0.5 year to the DR2 epoch. Output: SkyCoord object in DR2 epoch. Does not take into account parallax."""
    c = SkyCoord(ra = ra_g3 * u.deg, 
                 dec = dec_g3 * u.deg, 
                 distance = 1 * u.kpc, #setting distance to 1 kpc, otherwise it thinks stuff is at 10 Mpc and then returns an exception due to faster than light
                 pm_ra_cosdec = pmra_g3 * u.mas/u.yr,
                 pm_dec = pmdec_g3 * u.mas/u.yr,
                 obstime = Time(2016.0, format='jyear'))
    return c.apply_space_motion(Time(2015.5, format='jyear'))


# In[17]:


def get_norm(pairs_list):
    
    #Propagate back and add two new columns containing the calculated dr2 position
    
    z = propagate_back_linear(pairs_list['ra'].to_numpy(), pairs_list['dec'].to_numpy(), pairs_list['pmra'].to_numpy(), pairs_list['pmdec'].to_numpy())
    pairs_list['ra_2'] = z.ra.deg
    pairs_list['dec_2'] = z.dec.deg
    
    #List of conditions
    conditions = [
    (~pairs_list['phot_bp_mean_flux.1'].isna() & ~pairs_list['phot_rp_mean_flux.1'].isna()),
    
    (pairs_list['phot_bp_mean_flux.1'].isna() & ~pairs_list['phot_rp_mean_flux.1'].isna()),
    (~pairs_list['phot_bp_mean_flux.1'].isna() & pairs_list['phot_rp_mean_flux.1'].isna()),
    
    (pairs_list['phot_bp_mean_flux.1'].isna() & pairs_list['phot_rp_mean_flux.1'].isna()),
    ]

    ra_offset = (pairs_list['ra_2']-pairs_list['ra.1'])*np.cos(pairs_list['dec_2']*myU.degree)*myU.degree/myU.mas
    dec_offset = (pairs_list['dec_2']-pairs_list['dec.1'])*myU.degree/myU.mas
    
    #Contingent on each condition, evaluate the following normalized norm:
    norms = [
    (1/7)*(ra_offset**2/(pairs_list['ra_error']**2) + dec_offset**2/(pairs_list['dec_error']**2) + (pairs_list['pmra']-pairs_list['pmra.1'])**2/(pairs_list['pmra_error']**2) + (pairs_list['pmdec']-pairs_list['pmdec.1'])**2/(pairs_list['pmdec_error']**2) + (pairs_list['parallax']-pairs_list['parallax.1'])**2/(pairs_list['parallax_error']**2) + (pairs_list['phot_bp_mean_flux']-pairs_list['phot_bp_mean_flux.1'])**2/(pairs_list['phot_bp_mean_flux_error']**2) + (pairs_list['phot_rp_mean_flux']-pairs_list['phot_rp_mean_flux.1'])**2/(pairs_list['phot_rp_mean_flux_error']**2)),
    
    (1/6)*(ra_offset**2/(pairs_list['ra_error']**2) + dec_offset**2/(pairs_list['dec_error']**2) + (pairs_list['pmra']-pairs_list['pmra.1'])**2/(pairs_list['pmra_error']**2) + (pairs_list['pmdec']-pairs_list['pmdec.1'])**2/(pairs_list['pmdec_error']**2) + (pairs_list['parallax']-pairs_list['parallax.1'])**2/(pairs_list['parallax_error']**2)  + (pairs_list['phot_rp_mean_flux']-pairs_list['phot_rp_mean_flux.1'])**2/(pairs_list['phot_rp_mean_flux_error']**2)),
    (1/6)*(ra_offset**2/(pairs_list['ra_error']**2) + dec_offset**2/(pairs_list['dec_error']**2) + (pairs_list['pmra']-pairs_list['pmra.1'])**2/(pairs_list['pmra_error']**2) + (pairs_list['pmdec']-pairs_list['pmdec.1'])**2/(pairs_list['pmdec_error']**2) + (pairs_list['parallax']-pairs_list['parallax.1'])**2/(pairs_list['parallax_error']**2)  + (pairs_list['phot_bp_mean_flux']-pairs_list['phot_bp_mean_flux.1'])**2/(pairs_list['phot_bp_mean_flux_error']**2)),
    
    (1/5)*(ra_offset**2/(pairs_list['ra_error']**2) + dec_offset**2/(pairs_list['dec_error']**2) + (pairs_list['pmra']-pairs_list['pmra.1'])**2/(pairs_list['pmra_error']**2) + (pairs_list['pmdec']-pairs_list['pmdec.1'])**2/(pairs_list['pmdec_error']**2) + (pairs_list['parallax']-pairs_list['parallax.1'])**2/(pairs_list['parallax_error']**2)),
    
    ]

    pairs_list['norm'] = np.select(conditions, norms, default=False)
    return pairs_list


# In[18]:


def match_pairs(pairs_list1):
    pairs_list = get_norm(pairs_list1)
    #mask by condition norm < 4
    first_cut = pairs_list[pairs_list['norm']<4]
    first_cut.shape
    
    #Sort by source id, then norm. The duplicates with the smallest norm are at the top of their respective "chunk."
    first_cut.sort_values(['source_id', 'norm'], ascending = [True, True],inplace=True)
    
    #Drop all duplicates, keep the one with the smallest norm
    first_cut.drop_duplicates(subset=['source_id'],keep = 'first', inplace=True)
    
    #Do the same, except for dr2 source. This ensures that two different dr3 sources don't get matched to the same dr2 source.
    #Keep the one with the smallest norm
    first_cut.sort_values(['source_id.1', 'norm'], ascending = [True, True],inplace=True)
    first_cut.drop_duplicates(subset=['source_id.1'],keep = 'first', inplace=True)
    
    #Re-sort the dataframe by edr3 source id, for convenience
    first_cut.sort_values(['source_id', 'norm'], ascending = [True, True],inplace=True)
    
    return first_cut


# In[19]:


def fn_hacky_accel(th_2, th_3, mu_2, mu_3):
    """
    Function to compute the hacky acceleration vector. 
    Takes as inputs (N, 2) arrays for the DR2 position, eDR3 position, DR2, proper motion, eDR3 proper motion vectors. 
    Return (N, 2) array for the acceleration vectors (in mas/y^2).
    """
    t3 = 34.12/12
    t2 = 21.96/12

    tg3 = 17.26/12
    tg2 = 10.6849/12
    
    
    delta_th = np.array([(th_2[:, 0] - th_3[:, 0])*np.cos(th_3[:, 1]*myU.degree)*myU.degree/myU.mas, (th_2[:, 1] - th_3[:, 1])*myU.degree/myU.mas]).T
    delta_mu = mu_3*tg3 - mu_2*tg2
    
    acc_vec = 12*(delta_th + delta_mu)/(t3**2 - t2**2) 
    
    return acc_vec


# In[20]:


def generate_accel_cat(dr3, dr2):
    #Generate dataframe with dr3 matched with corresponding dr2 source
    pairs_list = generate_pairs_list(dr3,dr2)
    pair_df1 = get_norm(pairs_list)
    pair_df = match_pairs(pair_df1)
    
    #make (N,2) array of acceleration vectors (mas/y^2)
    th_2 = np.array(pair_df[['ra.1', 'dec.1']])
    th_3 = np.array(pair_df[['ra', 'dec']])
    mu_2 = np.array(pair_df[['pmra.1', 'pmdec.1']])
    mu_3 = np.array(pair_df[['pmra', 'pmdec']])

    accels = fn_hacky_accel(th_2, th_3, mu_2, mu_3)
    
    # Anonymous function to find the error of the hacky acceleration
    t3 = 34.12/12
    t2 = 21.96/12

    tg3 = 17.26/12
    tg2 = 10.6849/12
    
    hacky_error = lambda sig_th_2,sig_th_3,sig_mu_2,sig_mu_3 : 12*(np.sqrt(sig_th_2**2 + sig_th_3**2 + tg2**2*sig_mu_2**2 + tg3**2*sig_mu_3**2))/(t3**2 - t2**2)

    ra2_error = np.array(pair_df['ra_error.1'])
    dec2_error = np.array(pair_df['dec_error.1'])

    pmra2_error = np.array(pair_df['pmra_error.1'])
    pmdec2_error = np.array(pair_df['pmdec_error.1'])


    ra3_error = np.array(pair_df['ra_error'])
    dec3_error = np.array(pair_df['dec_error'])

    pmra3_error = np.array(pair_df['pmra_error'])
    pmdec3_error = np.array(pair_df['pmdec_error'])

    # Find acceleration errors
    accel_ra_error = hacky_error(ra2_error, ra3_error, pmra2_error, pmra3_error)
    accel_dec_error = hacky_error(dec2_error, dec3_error, pmdec2_error, pmdec3_error)    
    
    pair_df['accel_ra'] = accels[:,0]
    pair_df['accel_dec'] = accels[:,1]
    
    pair_df['accel_ra_error'] = accel_ra_error
    pair_df['accel_dec_error']= accel_dec_error
    
    pair_df.reset_index(inplace = True, drop=True)
    # Return minimal acceleration catalogue, with only source ID and accelerations + errors
    minimal = pair_df[['source_id','source_id.1','accel_ra', 'accel_ra_error', 'accel_dec', 'accel_dec_error']]
    minimal.columns = ['source_id_edr3', 'source_id_dr2','accel_ra', 'accel_ra_error', 'accel_dec', 'accel_dec_error']
    
    acc_stats_col = ['source_id', 'parallax', 'parallax_error', 'accel_ra', 'accel_dec', 
                     'accel_ra_error', 'accel_dec_error', 'phot_g_mean_mag']
    
    return minimal, pair_df[acc_stats_col]


# # Acceleration statistics

# In[21]:


nside = 2**8
fac_source_id = 2**(59-2*8)
npix = hp.nside2npix(nside)
#print('nside =',nside,', npix =',npix)
#print('linear pixel size =',str(np.sqrt(4*np.pi / npix) / arcsec)[0:7],' arcsec =', str(np.sqrt(4*np.pi / npix) / degree)[0:7],' degree')


# In[22]:


# bin definitions
bins_parallax = np.concatenate([[-1000],np.logspace(np.log10(0.05),np.log10(2),10),[1000]])
#print(bins_parallax)
bins_G = np.arange(3,23,1) # floor or the min and max G mag in the entire catalog are 3 and 21
#print(bins_G)


# In[23]:


def fn_acc_stats(tab, th_count=3, return_tab=False, n_sigma_out = 3): 
    """
    Bins the stars in tab in healpix, G mag and parallax and computes the mean and variance of acc_ra and acc_dec per bin.
    If return_tab=False, returns the statistic in each bin.
    If return_tab=True, returns the stars in tab removing the outliers at more than n_sigma_out from the mean.
    """

    ### healpix binning
    q_pix = np.floor(tab['source_id'].to_numpy() / fac_source_id).astype(int)
    bins_pix = np.arange(np.min(np.unique(q_pix)), np.max(np.unique(q_pix))+2,1) # should be +2 to include sources in the last bin
    q_binpix = np.digitize(q_pix, bins_pix)-1  # need to access the histogram matrix elements

    ### assign to G bins
    tab_G = tab['phot_g_mean_mag'].to_numpy()
    q_binG = np.digitize(tab_G, bins_G)-1      
    
    ### probabilistic assignment to parallax bins
    tab_parallax = tab['parallax'].to_numpy(); tab_parallax_error = tab['parallax_error'].to_numpy();
    prob_parallax = np.nan * np.ones((len(tab),len(bins_parallax)-1))
    for i in range(len(bins_parallax)-1):
        x1_list = (bins_parallax[i]-tab_parallax)/tab_parallax_error/np.sqrt(2)
        x2_list = (bins_parallax[i+1]-tab_parallax)/tab_parallax_error/np.sqrt(2)
        prob_parallax[:,i] = 0.5*(erf(x2_list)-erf(x1_list))

    tab_acc_ra = tab['accel_ra'].to_numpy(); tab_acc_dec = tab['accel_dec'].to_numpy();
    ### histogram of summed probabilities
    hist_prob = stats.binned_statistic_dd([tab_G,q_pix],np.transpose(prob_parallax), bins=[bins_G,bins_pix],statistic='sum')[0] 
    ### histogram of average acc_ra weighted by probabilities
    hist_acc_ra = stats.binned_statistic_dd([tab_G,q_pix],np.transpose(prob_parallax) * tab_acc_ra, bins=[bins_G,bins_pix],statistic='sum')[0] #sum first in each bin
    hist_acc_ra = hist_acc_ra / (hist_prob + 1e-20) #then divide by number in each bin
    hist_acc_dec = stats.binned_statistic_dd([tab_G,q_pix],np.transpose(prob_parallax) * tab_acc_dec, bins=[bins_G,bins_pix],statistic='sum')[0] #sum first in each bin
    hist_acc_dec = hist_acc_dec / (hist_prob + 1e-20) #then divide by number in each bin
    
    ### For each star, get the mean acc of the corresponding bin
    mean_acc_ra = hist_acc_ra[:, q_binG, q_binpix].T; mean_acc_dec = hist_acc_dec[:, q_binG, q_binpix].T

    ### histogram of acc variance weighted by parallax bin probabilities
    hist_acc_ra_var = stats.binned_statistic_dd([tab_G,q_pix],np.transpose(prob_parallax) * (mean_acc_ra.T - tab_acc_ra)**2,
                                                   bins=[bins_G,bins_pix],statistic='sum')[0] #sum first in each bin
    hist_acc_ra_var = hist_acc_ra_var / (hist_prob - 1 + 1e-20) # the estimator should have a -1 (this matches for example var() computed with panda's groupy)
    hist_acc_dec_var = stats.binned_statistic_dd([tab_G,q_pix],np.transpose(prob_parallax) * (mean_acc_dec.T - tab_acc_dec)**2,
                                                    bins=[bins_G,bins_pix],statistic='sum')[0] #sum first in each bin
    hist_acc_dec_var = hist_acc_dec_var / (hist_prob - 1 + 1e-20) 
    hist_acc_radec_var = stats.binned_statistic_dd([tab_G,q_pix],np.transpose(prob_parallax) * (mean_acc_ra.T - tab_acc_ra) * (mean_acc_dec.T - tab_acc_dec),
                                                      bins=[bins_G,bins_pix],statistic='sum')[0] #sum first in each bin
    hist_acc_radec_var = hist_acc_radec_var / (hist_prob - 1 + 1e-20) 
    
    ### set to nan bins where there are too few stars
    hist_acc_ra[hist_prob < th_count] = np.nan; hist_acc_dec[hist_prob < th_count] = np.nan
    hist_acc_ra_var[hist_prob < th_count] = np.nan; hist_acc_dec_var[hist_prob < th_count] = np.nan; hist_acc_radec_var[hist_prob < th_count] = np.nan

    if return_tab==False: # returns the data frame with the statistics computed using tab
        ###  filler for generalized bins indices
        hist_bins_pix = np.ones(np.shape(hist_prob)) * bins_pix[:-1]
        hist_bins_G = np.transpose(np.transpose(np.ones(np.shape(hist_prob)),axes=[0,2,1]) * bins_G[:-1],axes=[0,2,1])
        hist_bins_parallax = np.transpose(np.transpose(np.ones(np.shape(hist_prob)),axes=[2,1,0]) * bins_parallax[:-1],axes=[2,1,0])

        ###  collect data and output
        data = np.transpose([hist_bins_pix, hist_bins_G, hist_bins_parallax, hist_prob, hist_acc_ra, hist_acc_dec, hist_acc_ra_var, hist_acc_dec_var, hist_acc_radec_var],axes=[1,2,3,0])
        data = data.reshape(-1, data.shape[-1])
        return pd.DataFrame(data,columns=['pix','G_bin','parallax_bin','number','mean_acc_ra','mean_acc_dec','var_acc_ra','var_acc_dec','var_acc_radec'])
    
    else: # returns tab where the acc outliers more than n_sigma_out away from zero have been removed
        ### For each star, get the acc mean and variance of the corresponding bin (after excluding the low count bins)
        mean_acc_ra = hist_acc_ra[:, q_binG, q_binpix].T; mean_acc_dec = hist_acc_dec[:, q_binG, q_binpix].T
        var_acc_ra = hist_acc_ra_var[:, q_binG, q_binpix].T; var_acc_dec = hist_acc_dec_var[:, q_binG, q_binpix].T; var_acc_radec = hist_acc_radec_var[:, q_binG, q_binpix].T;    

        ###  Get the mean and var for each star
        tab_sum_pw = np.sum(prob_parallax, axis=1, where=(~np.isnan(mean_acc_ra)))  # sum of the parallax weights for each star using only bins with enough statistics 
        tab_mean_acc_ra = np.sum(np.nan_to_num(mean_acc_ra*prob_parallax), axis=1)/(tab_sum_pw + 1e-20)
        tab_mean_acc_dec = np.sum(np.nan_to_num(mean_acc_dec*prob_parallax), axis=1)/(tab_sum_pw + 1e-20)
        tab_var_acc_ra = np.sum(np.nan_to_num(var_acc_ra*prob_parallax), axis=1)/(tab_sum_pw + 1e-20)
        tab_var_acc_dec = np.sum(np.nan_to_num(var_acc_dec*prob_parallax), axis=1)/(tab_sum_pw + 1e-20)
        tab_var_acc_radec = np.sum(np.nan_to_num(var_acc_radec*prob_parallax), axis=1)/(tab_sum_pw + 1e-20)        
        
        ### Replace the effective variance with the measurement errors for stars that have 0 mean (fall into empty bins)
        tab_var_acc_ra[tab_var_acc_ra==0] = (tab['accel_ra_error'].to_numpy()[tab_var_acc_ra==0])**2
        tab_var_acc_dec[tab_var_acc_dec==0] = (tab['accel_dec_error'].to_numpy()[tab_var_acc_dec==0])**2
        tab_var_acc_radec[tab_var_acc_radec==0] = (np.zeros(len(tab))*tab['accel_ra_error'].to_numpy()*tab['accel_dec_error'].to_numpy())[tab_var_acc_radec==0]
        
        ### subtracted acc and inverse covariance for outlier removal
        acc_sub = np.array([tab['accel_ra'].to_numpy()-tab_mean_acc_ra, tab['accel_dec'].to_numpy()-tab_mean_acc_dec]).T
        inv_cov_acc = np.linalg.inv(np.array([[tab_var_acc_ra, tab_var_acc_radec], [tab_var_acc_radec, tab_var_acc_dec]]).T)
        acc_over_sigma_sq = inv_cov_acc[:, 0, 0]*acc_sub[:, 0]**2 + inv_cov_acc[:, 1, 1]*acc_sub[:, 1]**2 + 2*inv_cov_acc[:, 0, 1]*acc_sub[:, 0]*acc_sub[:, 1]
        
        return tab.iloc[acc_over_sigma_sq < n_sigma_out**2]


# In[24]:


def fn_execute(tab, i_f, n_iter=10, flag_print=False):
        
    tab = tab.loc[tab['phot_g_mean_mag']>0] # only look at stars with pm and G > 0

    i=0; out_frac=1
    while (i<n_iter) & (out_frac>1E-5):
        tab_temp = fn_acc_stats(tab, th_count=3, return_tab=True, n_sigma_out = 3)
        i+=1; out_frac=(1-len(tab_temp)/len(tab)); 
        if flag_print==True:
            print('Iter '+str(i)+' -- fraction of outliers removed: '+str(out_frac*100)[:7]+' %'); sys.stdout.flush()
        tab = tab_temp 
        
    df_acc_stats = fn_acc_stats(tab, return_tab=False) 
    df_acc_stats.to_csv(hist_res_dir+list_dr3_files[i_f][:-7]+'_hist.csv', index=False) #write to file
        
    return len(tab)


# # Make Both Catalogues

# In[25]:


pair_cat = generate_pair_cat(dr3, 3, 2) #cutoff at 3 arcsec, 95% CL


# In[26]:


accel_cat, tab_acc_stat = generate_accel_cat(dr3, dr2)


# ### When possible, include acceleration of background source in pair_cat

# In[27]:


def add_accel_cols(pair_cat, accel_cat):
    #When possible, this function adds acceleration data to the column in 
    indices = np.intersect1d(pair_cat['source_id_bg'],accel_cat['source_id_edr3'], return_indices = True)
    
    accel_ra_to_add = np.empty(len(pair_cat))
    accel_dec_to_add = np.empty(len(pair_cat))
    
    accel_ra_err_to_add = np.empty(len(pair_cat))
    accel_dec_err_to_add = np.empty(len(pair_cat))
    
    #Initialize accelerations with NaN values
    accel_ra_to_add[:] = np.nan
    accel_dec_to_add[:] = np.nan
    
    accel_ra_err_to_add[:] = np.nan
    accel_dec_err_to_add[:] = np.nan
    
    for i in range(len(indices[1])):
        accel_ra_to_add[indices[1][i]] = accel_cat['accel_ra'].iloc[indices[2][i]]
        accel_dec_to_add[indices[1][i]] = accel_cat['accel_dec'].iloc[indices[2][i]]
        
        accel_ra_err_to_add[indices[1][i]] = accel_cat['accel_ra_error'].iloc[indices[2][i]]
        accel_dec_err_to_add[indices[1][i]] = accel_cat['accel_dec_error'].iloc[indices[2][i]]
        
    pair_cat['accel_ra_bg'] = accel_ra_to_add
    pair_cat['accel_ra_error_bg'] = accel_ra_err_to_add
    
    pair_cat['accel_dec_bg'] = accel_dec_to_add
    pair_cat['accel_dec_error_bg'] = accel_dec_err_to_add


# In[28]:


add_accel_cols(pair_cat, accel_cat)


# # Export catalogues to CSV

# In[554]:


accel_cat_name = 'accels_' + str(healpix_edr3_start[current_index]) +'-'+ str(healpix_edr3_end[current_index])
accel_cat.to_csv('./acceleration_catalogue/'+ accel_cat_name)


# In[555]:


pair_cat_name = 'pairs_' + str(healpix_edr3_start[current_index]) +'-'+ str(healpix_edr3_end[current_index])
pair_cat.to_csv('./accidental_pairs/' + pair_cat_name)


# # Compute acceleration statistics and save histograms

# In[32]:


fn_execute(tab_acc_stat, current_index, n_iter=10, flag_print=False)


# In[ ]:




