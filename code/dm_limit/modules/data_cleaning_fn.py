import numpy as np
import healpy as hp
import math
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.interpolate import griddata
from tqdm import tqdm, tqdm_notebook
from my_units import *
from angular_fn import *

def fn_nb_pixel(patch_pix, radius_nb, nside, nest=True):
    """
    For healpy pixels in the 1d array patch_pix, returns the neighbors of each pixel within a disc of radius radius_nb (in rad).
    Notice that search_around_sky is not faster than this function if we need to query neighbors within 0.1-0.3 degree. Healpy pixelation is more efficient in that case.
    """    
    ### Cartesian vectors for each pixel in patch_pix
    vec_pix_x, vec_pix_y, vec_pix_z = hp.pix2vec(nside, patch_pix, nest=nest) 
    vec_array = np.array([vec_pix_x, vec_pix_y, vec_pix_z]).T
    
    ### Loop over pixels
    nb_pix = []; 
    for i in range(len(patch_pix)):
        ### Disc around the pixel position  
        nb_pix.append(hp.query_disc(nside, vec_array[i], radius_nb, inclusive=True, nest=nest))      
    return nb_pix


def fn_remove_clumps(data, disc_center, disc_radius, beta_kernel=0.1*degree, f_clumps=3):
    """
    Removes stars falling in overdense regions, where the density is f_clumps times the local density field computed with a gaussian kernel of size beta_kernel.
    """
    
    ### Determine pixelation scale (approx beta_kernel/8)
    n = round(math.log(np.sqrt(np.pi/3)/(beta_kernel/8), 2))
    #n = round(math.log(np.sqrt(np.pi/3)/(beta_kernel/10), 2))
    nside = 2**n; npix = hp.nside2npix(nside);
    pix_size = np.sqrt(4*np.pi / npix) / degree
    print('Linear pixel size = ', pix_size, ' degree')

    vec = hp.pix2vec(nside, hp.ang2pix(nside, disc_center[0], disc_center[1], nest=True, lonlat=True), nest=True)
    disc_pix = hp.query_disc(nside, vec, disc_radius, nest=True, inclusive=True) # select only pixels on the sky within the selected disc
    n_disc_pix = len(disc_pix)
    disc_pix_ra, disc_pix_dec = hp.pix2ang(nside, disc_pix, nest=True, lonlat=True) # coordinates of the selected pixels
    
    ### Pixelate stars using dataframe groupby
    q_pix = np.asarray(hp.ang2pix(nside, data['ra'].to_numpy(), data['dec'].to_numpy(), nest=True, lonlat=True)) # healpy pixel number of the stars
    df_hist = pd.DataFrame({'q_pix_{}'.format(n):q_pix, 'ra':data['ra'].to_numpy(), 'dec':data['dec'].to_numpy()}).groupby(by=['q_pix_{}'.format(n)], as_index=False).sum()
    occ_pix = df_hist['q_pix_{}'.format(n)].to_numpy() # uniquely occupied pixels
    pix_count = np.bincount(q_pix); # number of stars per pixel  
    filled_pix_count = pix_count[pix_count>0]

    ### Density and coordinates per pixel
    all_density = np.zeros(npix); all_density[occ_pix] = filled_pix_count/pix_size**2
    all_mean_coord = np.zeros((npix, 2)); 
    all_mean_coord[disc_pix] = np.array([disc_pix_ra, disc_pix_dec]).T # set pix coordinates to the pix center
    all_mean_coord[occ_pix] = np.array([df_hist['ra'].to_numpy()/filled_pix_count, df_hist['dec'].to_numpy()/filled_pix_count]).T # set pix coordinates to the mean coordinate for non empy pixels
    
    ### Find neighboring pixels for each pixel
    radius_nb = 2.5*beta_kernel; # radius of neighboring pixels
    vec_pix_x, vec_pix_y, vec_pix_z = hp.pix2vec(nside, disc_pix, nest=True) 
    vec_array = np.array([vec_pix_x, vec_pix_y, vec_pix_z]).T # crtesian vectors for each pixel in patch_pix
    
    ### Loop over pixels
    density_gauss = np.zeros(n_disc_pix)
    for i in tqdm(range(n_disc_pix)):
        nb_pix = hp.query_disc(nside, vec_array[i], radius_nb, inclusive=True, nest=True)
        rel_distance_sq = fn_angular_sep_magn_sq(all_mean_coord[disc_pix[i], 0]*degree, all_mean_coord[disc_pix[i], 1]*degree, 
                                                 all_mean_coord[nb_pix, 0]*degree, all_mean_coord[nb_pix, 1]*degree)/(2*beta_kernel**2) # distance from the pixel
        gauss_weights = np.exp(-rel_distance_sq); 
        density_gauss[i] = sum(all_density[nb_pix]*gauss_weights)/sum(gauss_weights)  # gaussian weighted mean density 

    overdense_pixels = disc_pix[all_density[disc_pix] >= f_clumps*density_gauss]
    
    return data[~np.isin(q_pix, overdense_pixels)], hp.pix2ang(nside, overdense_pixels, nest=True, lonlat=True)


def fn_prepare_back_sub(data, disc_center, disc_radius, beta_kernel_sub):
    """
    Prepare the data for the background motion subtraction.
    """
    ### Pixelation at approx 1/3 of beta_kernel
    n = round(math.log(np.sqrt(np.pi/3)/(beta_kernel_sub/3), 2))   
    nside = 2**n; npix = hp.nside2npix(nside);

    vec = hp.pix2vec(nside, hp.ang2pix(nside, disc_center[0], disc_center[1], nest=True, lonlat=True), nest=True)
    disc_pix = hp.query_disc(nside, vec, disc_radius, nest=True, inclusive=True) # pixels on the sky within the selected disc
    
    ### Stars healpy pixel number
    q_pix = np.asarray(hp.ang2pix(nside, data['ra'].to_numpy(), data['dec'].to_numpy(), nest=True, lonlat=True)) # healpy pixel number of the stars
    data.loc[:, ('q_pix_{}'.format(n))] = q_pix    
    
    ### Find neighboring pixels for each pixel
    nb_pixel_list = fn_nb_pixel(disc_pix, 3*beta_kernel_sub, nside, nest=True)

    return disc_pix, nb_pixel_list, n


def fn_back_field_sub(data, disc_pix, nb_pixel_array, n, beta_kernel=0.1*degree, sub=False, sim=False):
    """
    Creates a local map of the pm field using a gaussian distance kenerl of size beta_kernel and subtracts the mean fields from each star pm.
    If sub=True, the subtracted proper motions from a previous iteration are used.
    If sim=True, the simulated proper motions are used.
    """
    
    nside = 2**n; npix = hp.nside2npix(nside);
    
    ### Pixelate stars using dataframe groupby
    if (sub==False) & (sim==False):
        old_pmra = data['pmra'].to_numpy(); old_pmdec = data['pmdec'].to_numpy(); 
    elif (sub==True) & (sim==False):
        old_pmra = data['pmra_sub'].to_numpy(); old_pmdec = data['pmdec_sub'].to_numpy(); 
        data.drop(labels=['pmra_sub', 'pmdec_sub'], axis="columns", inplace=True) 
    elif (sub==False) & (sim==True):
        old_pmra = data['pmra_sim'].to_numpy(); old_pmdec = data['pmdec_sim'].to_numpy(); 
        data.drop(labels=['pmra_sim', 'pmdec_sim'], axis="columns", inplace=True) 

    df_hist = pd.DataFrame({'q_pix_{}'.format(n):data['q_pix_{}'.format(n)].to_numpy(), 
                            'ra':data['ra'].to_numpy(), 'dec':data['dec'].to_numpy(),
                            'weighted_pmra':old_pmra/data['pmra_error'].to_numpy()**2, 
                            'weighted_pmdec':old_pmdec/data['pmdec_error'].to_numpy()**2, 
                            'pmra_w':1/data['pmra_error'].to_numpy()**2, 'pmdec_w':1/data['pmdec_error'].to_numpy()**2}).groupby(by=['q_pix_{}'.format(n)], as_index=False).sum()
            
    occ_pix = df_hist['q_pix_{}'.format(n)].to_numpy() # uniquely occupied pixels
    pix_count = np.bincount(data['q_pix_{}'.format(n)].to_numpy()); # number of stars per pixel  
    filled_pix_count = pix_count[pix_count>0]
    
    ### Full sky pixel arrays
    disc_pix_ra, disc_pix_dec = hp.pix2ang(nside, disc_pix, nest=True, lonlat=True) # coordinates of the selected pixels
    all_mean_coord = np.zeros((npix, 2)); 
    all_mean_coord[disc_pix] = np.array([disc_pix_ra, disc_pix_dec]).T # set pix coordinates to the pix center
    all_mean_coord[occ_pix] = np.array([df_hist['ra'].to_numpy()/filled_pix_count, df_hist['dec'].to_numpy()/filled_pix_count]).T # set pix coordinates to the mean coordinate

    all_mean_pm = np.zeros((npix, 2)); 
    all_mean_pm[occ_pix] = np.array([df_hist['weighted_pmra'].to_numpy()/df_hist['pmra_w'].to_numpy(), 
                                     df_hist['weighted_pmdec'].to_numpy()/df_hist['pmdec_w'].to_numpy()]).T    
    
    n_disc_pix = len(disc_pix)
    pm_gauss = np.zeros((n_disc_pix, 2)); 
    
    ### Loop over pixels
    for i in range(n_disc_pix):
        nb_pix = nb_pixel_array[i]     
        rel_distance_sq = fn_angular_sep_magn_sq(all_mean_coord[disc_pix[i], 0]*degree, all_mean_coord[disc_pix[i], 1]*degree, 
                                                 all_mean_coord[nb_pix, 0]*degree, all_mean_coord[nb_pix, 1]*degree)/(2*beta_kernel**2)
        gauss_weights = np.exp(-rel_distance_sq); sum_gauss_weights = sum(gauss_weights)
        pm_gauss[i, 0] = sum(all_mean_pm[nb_pix, 0]*gauss_weights)/sum_gauss_weights  # gaussian weighted mean pm in ra
        pm_gauss[i, 1] = sum(all_mean_pm[nb_pix, 1]*gauss_weights)/sum_gauss_weights  # gaussian weighted mean pm in dec
        
    ### Interpolation of the velocity field
    pmra_interp = griddata((all_mean_coord[disc_pix, 0], all_mean_coord[disc_pix, 1]), pm_gauss[:, 0], (data['ra'].to_numpy(), data['dec'].to_numpy()), method='linear', fill_value=0)
    pmdec_interp = griddata((all_mean_coord[disc_pix, 0], all_mean_coord[disc_pix, 1]), pm_gauss[:, 1], (data['ra'].to_numpy(), data['dec'].to_numpy()), method='linear', fill_value=0)
    
    if sim==False:
        data.insert(len(data.columns), 'pmra_sub', old_pmra - pmra_interp); data.insert(len(data.columns), 'pmdec_sub', old_pmdec - pmdec_interp)
    else:
        data.insert(len(data.columns), 'pmra_sim', old_pmra - pmra_interp); data.insert(len(data.columns), 'pmdec_sim', old_pmdec - pmdec_interp)
    
    return None


def fn_rem_outliers(data, pm_esc, D_s, n_sigma_out=3, sim=False):
    """
    Remove stars with pm more than n_sigma_out sigma away from the expected value
    Returns cleaned stars and fraction of outliers removed
    """
    old_len = len(data)
    if sim==False:
        new_data = data[( np.sqrt(data['pmra_sub'].to_numpy()**2 + data['pmdec_sub'].to_numpy()**2) < 
                          (pm_esc + n_sigma_out*np.sqrt(data['pmra_error'].to_numpy()**2 + data['pmdec_error'].to_numpy()**2)) ) ]
    else:
        new_data = data[( np.sqrt(data['pmra_sim'].to_numpy()**2 + data['pmdec_sim'].to_numpy()**2) < 
                          (pm_esc + n_sigma_out*np.sqrt(data['pmra_error'].to_numpy()**2 + data['pmdec_error'].to_numpy()**2)) ) ]   
    return new_data, 1-len(new_data)/old_len


def fn_rem_edges(data, disc_center, disc_radius):
    """
    Keep only stars within disc_radius of the disc_center, to remove the edges.
    """
    center_sky_coord = SkyCoord(ra = disc_center[0] * u.deg, dec = disc_center[1] * u.deg)
    data_sky_coord = SkyCoord(ra = data['ra'].to_numpy() * u.deg, dec = data['dec'].to_numpy() * u.deg)
    data_r = data_sky_coord.separation(center_sky_coord).value*degree
     
    return data[data_r < disc_radius]    


def fn_effective_w(data, disc_center, gmag_bin_size=0.1, rad_bin_size=1, sim=False):
    """
    Computethe  effective pm dispersions in G magnitude and radial bins. 
    Adds the effective error column to the data, taking the max between the instrumental and effective pm dispersion.
    """
     
    ### Bin in g magnitude and radial distance from the center
    data_g = data['phot_g_mean_mag'].to_numpy()
    max_g = np.max(data_g)
    n_bins_g = int(math.ceil(max_g/gmag_bin_size))
    bins_g = np.linspace(0, gmag_bin_size*n_bins_g, n_bins_g+1)
    q_bin_g = np.digitize(data_g, bins_g)-1  
    
    center_sky_coord = SkyCoord(ra = disc_center[0] * u.deg, dec = disc_center[1] * u.deg)
    data_sky_coord = SkyCoord(ra = data['ra'].to_numpy() * u.deg, dec = data['dec'].to_numpy() * u.deg)
    data_r = data_sky_coord.separation(center_sky_coord).value
    bins_r = np.arange(0, np.max(data_r)+rad_bin_size, rad_bin_size)
    q_bin_r = np.digitize(data_r, bins_r)-1     
    
    ### Histograms with mean pm dispersion per bin
    counts = np.histogram2d(data_g, data_r, bins=[bins_g, bins_r], weights=None)[0]
    if sim==False:
        pm_sq = np.histogram2d(data_g, data_r, bins=[bins_g, bins_r], weights=(data['pmra_sub'].to_numpy()**2 + data['pmdec_sub'].to_numpy()**2))[0]
    else:
        pm_sq = np.histogram2d(data_g, data_r, bins=[bins_g, bins_r], weights=(data['pmra_sim'].to_numpy()**2 + data['pmdec_sim'].to_numpy()**2))[0]

    sigma_pm_eff_hist = np.sqrt(np.divide(pm_sq, counts, out=np.zeros_like(pm_sq), where=counts!=0))
    
    ### Add effective error column (for each star, take the max between the instrumental and effective dispersion)
    data.insert(len(data.columns), 'pm_eff_error', np.max(np.array([sigma_pm_eff_hist[q_bin_g, q_bin_r], np.sqrt(data['pmra_error'].to_numpy()**2 + data['pmdec_error'].to_numpy()**2)]), axis=0))
    
    return None
    

