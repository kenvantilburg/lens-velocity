#!/usr/bin/env python
# coding: utf-8

# In[533]:


import numpy as np
import myUnitsCopy1 as myU # customized library for units. All dimensional variables are in GeV and GeV=1
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.coordinates import *
import astropy.units as u

from os import listdir
import gzip
import sys


# # Loading in the Files

# ### Load a Single EDR3 File

# In[534]:


edr3_data = './edr3_data'
dr2_data = './dr2_data'


# In[535]:


list_dr3_files = listdir(edr3_data)


# In[536]:


healpix_edr3_start = np.empty((len(list_dr3_files)),dtype= int)
healpix_edr3_end = np.empty((len(list_dr3_files)), dtype = int)

for i,file in enumerate(list_dr3_files):
    int_1 = int(file[11:17])
    int_2 = int(file[18:24])
    healpix_edr3_start[i] = int_1
    healpix_edr3_end[i] = int_2
    


# In[537]:


def get_source_ids(file_names):
    #given a list of EDR3 filenames, return the start and end source IDs corresponding to healpix level 12
    N_8 = 2**(59-16)
    
    start = np.array([x*N_8 for x in healpix_edr3_start], dtype = 'int')
    end = np.array([x*N_8 for x in healpix_edr3_end], dtype = 'int')
    return start, end


# In[538]:


def load_dr3_file(idx):
    return pd.read_csv(edr3_data + '/' + list_dr3_files[idx], compression = 'gzip')


# In[539]:


start, end = get_source_ids(list_dr3_files)


# ### Load Corresponding DR2 files

# In[540]:


list_dr2_files = np.array([file for file in listdir(dr2_data) if file[-7:]=='.csv.gz']) #select only files ending with 'csv.gz'


# In[541]:


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

# In[ ]:


### Read in the eDR3 file index from the command line
current_index = int(sys.argv[1]) # current index in list of edr3 files
print('\nReading in eDR3 file '+str(current_index)+'.'); sys.stdout.flush()


# In[542]:


dr3 = load_dr3_file(current_index)
dr2 = load_dr2_files(current_index)


# # Generate Pair Catalogue
# 
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

# In[543]:


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

# In[544]:


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


# In[545]:


def propagate_back_linear(ra_g3, dec_g3, pmra_g3, pmdec_g3):
    """Takes EDR3 position and proper motion, and linearly propagates it by 0.5 year to the DR2 epoch. Output: SkyCoord object in DR2 epoch. Does not take into account parallax."""
    c = SkyCoord(ra = ra_g3 * u.deg, 
                 dec = dec_g3 * u.deg, 
                 distance = 1 * u.kpc, #setting distance to 1 kpc, otherwise it thinks stuff is at 10 Mpc and then returns an exception due to faster than light
                 pm_ra_cosdec = pmra_g3 * u.mas/u.yr,
                 pm_dec = pmdec_g3 * u.mas/u.yr,
                 obstime = Time(2016.0, format='jyear'))
    return c.apply_space_motion(Time(2015.5, format='jyear'))


# In[546]:


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


# In[547]:


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


# In[548]:


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


# In[549]:


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
    return minimal


# # Make Both Catalogues

# In[550]:


pair_cat = generate_pair_cat(dr3, 3, 2) #cutoff at 3 arcsec, 95% CL


# In[551]:


accel_cat = generate_accel_cat(dr3, dr2)


# ### When possible, include acceleration of background source in pair_cat

# In[552]:


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


# In[553]:


add_accel_cols(pair_cat, accel_cat)


# # Export to CSV

# In[554]:


accel_cat_name = 'accels_' + str(healpix_edr3_start[current_index]) +'-'+ str(healpix_edr3_end[current_index])
accel_cat.to_csv('./acceleration_catalogue/'+ accel_cat_name)


# In[555]:


pair_cat_name = 'pairs_' + str(healpix_edr3_start[current_index]) +'-'+ str(healpix_edr3_end[current_index])
pair_cat.to_csv('./accidental_pairs/' + pair_cat_name)

