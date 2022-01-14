import numpy as np
import healpy as hp
import math
import pandas as pd
from scipy import interpolate
import os
from my_units import *
from angular_fn import *

HomeDir = '../'
ListDir = HomeDir+'lists/'


def fn_coarse_scan_coord(disc_center, disc_radius, beta_t, n_betat=3.5):
    """
    Find coordinates for the coarse template scanning. Scan performed at angular scale approximatly beta_t.
    Works for a stellar target which is a circle on the sky.
    """
    ### Coarse pixelation of size approx. beta_t
    n = math.ceil(math.log(np.sqrt(np.pi/3)/beta_t, 2)); nside = 2**n; 
    vec = hp.pix2vec(nside, hp.ang2pix(nside, disc_center[0], disc_center[1], nest=True, lonlat=True), nest=True)
    disc_pix_coarse = hp.query_disc(nside, vec, disc_radius - n_betat*beta_t, nest=True, inclusive=False) ### pixels on the sky within a disc without the edge
    scan_coord = hp.pix2ang(nside, disc_pix_coarse, nest=True, lonlat=True) ### coordinates of the pixels center
    
    return scan_coord


def fn_prepare_template_scan(data, scan_coord, beta_t):
    """
    Prepare the data for the template scan. Pixelate at an angular scale approximatly beta_t/10. 
    Returns the pixel number of the scan_coord locations and the data.
    """    
    ### Fine pixelation of size approx. beta_t/10 (can be a bit larger than beta_t/10, so using round is fine)
    n = round(math.log(np.sqrt(np.pi/3)/(0.1*beta_t), 2)); nside = 2**n; 
    scan_pix = hp.ang2pix(nside, scan_coord[0], scan_coord[1], nest=True, lonlat=True) # healpy pixel number of the scan coordinates              
    data_pix = hp.ang2pix(nside, data[:, 0], data[:, 1], nest=True, lonlat=True) # healpy pixel number of the stars
    
    return nside, scan_pix, np.array([data_pix] + [data[:, i] for i in range(len(data[0]))]).T

def fn_prepare_template_scan_fine(data, scan_coord, beta_t, large_tau):
    """
    Prepare the data for the fine template scan. Pixelate at an angular scale approximatly beta_t/10. 
    Returns the pixel number of the scan_coord locations and the data.
    """      
    ### Fine pixelation of size approx. beta_t/10 (can be a bit larger than beta_t/10, so using round is fine)
    n = round(math.log(np.sqrt(np.pi/3)/(0.1*beta_t), 2)); nside = 2**n; 

    ### Pixels where to do the fine scanning
    fine_scan_pix = np.unique(hp.ang2pix(nside, scan_coord[0], scan_coord[1], nest=True, lonlat=True)) 
    large_tau_pix = hp.ang2pix(nside, large_tau[:, 0], large_tau[:, 1], nest=True, lonlat=True)
    
    fine_scan_pix = fine_scan_pix[~np.in1d(fine_scan_pix, large_tau_pix)]
    
    data_pix = hp.ang2pix(nside, data[:, 0], data[:, 1], nest=True, lonlat=True) # healpy pixel number of the stars
    
    return nside, fine_scan_pix, np.array([data_pix] + [data[:, i] for i in range(len(data[0]))]).T

# Uploading lists for the G_0 function and it's derivative. G_0 is the enclosed lens mass within a cylinder oriented along the line of sight. 
# See Eq.(3.10)=(3.11) of 1804.01991 or Eq.(2) of 2002.01938
# For the NFW truncated lens profile given by Eq.(3) of 2002.01938 the enclosed mass cannot be computed analytically. We use an interpolation function.

logxG0_list = np.loadtxt(ListDir+'G0NFWtrunc.txt');  logxG0_prime_list = np.loadtxt(ListDir+'G0pNFWtrunc.txt');  
logG0_fnc = interpolate.interp1d(logxG0_list[:, 0], logxG0_list[:, 1], kind='cubic', bounds_error=False, fill_value=(logxG0_list[0, 1], logxG0_list[-1, 1]))
logG0_p_fnc = interpolate.interp1d(logxG0_prime_list[:, 0], logxG0_prime_list[:, 1], kind='cubic', bounds_error=False, fill_value=(logxG0_prime_list[0, 1], logxG0_prime_list[-1, 1]))


### Returns the lens enclosed mass within the distance x = beta/beta_l
def G0_fnc(x): return np.power(10, logG0_fnc(np.log10(x+1E-20)))
def G0_p_fnc(x): return np.power(10, logG0_p_fnc(np.log10(x+1E-20)))

def fn_dipole_mf(b_l, b_vec):
    """
    Returns the proper motion dipole-like profile and the proper motion monopole profile.
    """
    b_norm = np.sqrt(b_vec[:, 0]**2 + b_vec[:, 1]**2)
    b_hat = np.array([b_vec[:,0]/(b_norm+1E-20), b_vec[:,1]/(b_norm+1E-20)]).T; x = b_norm/b_l
    G0_over_xsq = G0_fnc(x)/(x**2+1E-20); G0p_over_x = G0_p_fnc(x)/(x+1E-20)

    remove_inf = np.heaviside(b_norm, 0) ### set to zero values corresponding to b_vec = [0, 0], remove infinity at the origin
    
    dipole_ra = np.array([(G0_over_xsq*(2*b_hat[:, 0]*b_hat[:, 0] - 1) - G0p_over_x*b_hat[:,0]*b_hat[:,0])*remove_inf, 
                          (G0_over_xsq*(2*b_hat[:, 1]*b_hat[:, 0]) - G0p_over_x*b_hat[:, 1]*b_hat[:, 0])*remove_inf]).T
    dipole_dec = np.array([(G0_over_xsq*(2*b_hat[:, 0]*b_hat[:, 1]) - G0p_over_x*b_hat[:, 1]*b_hat[:, 0])*remove_inf, 
                           (G0_over_xsq*(2*b_hat[:, 1]*b_hat[:, 1] - 1) - G0p_over_x*b_hat[:, 1]*b_hat[:, 1])*remove_inf]).T
    isotropic_dipole_magn = (G0_over_xsq**2 + 0.5*(G0p_over_x**2-2*G0_over_xsq*G0p_over_x))*remove_inf # for the normalization; see Eq. (13) of 2002.01938
    
    ### Compute the monopole to check the background
    monopole = np.array([(G0_over_xsq*b_hat[:, 0])*remove_inf, (G0_over_xsq*b_hat[:, 1])*remove_inf]).T
    monopole_magn = G0_over_xsq**2*remove_inf
        
    return dipole_ra, dipole_dec, isotropic_dipole_magn, monopole, monopole_magn


def fn_template_scan(nside, scan_pix, n_betat, beta_t, data_np, min_mask=0.01*degree):
    """
    Computes the template at the locations given by coarse_scan_pix.
    Includes proper motion dipole, proper motion monopole and and parallax templates.
    """    
    scan_coord = hp.pix2ang(nside, scan_pix, nest=True, lonlat=True) # coordinates of the template locations, center of the pixel

    ### Cartesian vectors for each pixel in scan_pix
    vec_pix_x, vec_pix_y, vec_pix_z = hp.pix2vec(nside, scan_pix, nest=True) 
    vec_array = np.array([vec_pix_x, vec_pix_y, vec_pix_z]).T
    
    ### Quantities from the data needed to compute the template
    [data_q_pix, data_ra, data_dec, pm_w_sq, weighted_pmra, weighted_pmdec] = data_np.T
    
    n_loc = len(scan_pix)
    tau_mu_ra, tau_mu_dec, tau_mu_norm = np.zeros(n_loc), np.zeros(n_loc), np.zeros(n_loc)
    tau_mu_mon, tau_mu_mon_norm = np.zeros(n_loc), np.zeros(n_loc)    
    
    nb_radius = max(n_betat*beta_t, min_mask)
    
    for i in range(n_loc):
        nb_pix_i = hp.query_disc(nside, vec_array[i], nb_radius, inclusive=True, nest=True) ### disc around the template position 

        stars_in = ((data_q_pix >= nb_pix_i[0]) & (data_q_pix <= nb_pix_i[-1])) ### first reduce the total number of stars   
        nb_stars = np.isin(data_q_pix[stars_in], nb_pix_i, assume_unique=False, invert=False) ### keep only stars within the neighboring pixels 

        ### Pm template
        beta_it = fn_angular_sep(scan_coord[0][i]*degree, scan_coord[1][i]*degree, data_ra[stars_in][nb_stars]*degree, data_dec[stars_in][nb_stars]*degree)
        mu_ra, mu_dec, mu_sq, mu_mon, mu_mon_sq = fn_dipole_mf(beta_t, beta_it)

        tau_mu_norm[i] = np.sqrt(sum(mu_sq/pm_w_sq[stars_in][nb_stars])) ## normalization
        tau_mu_ra[i] = sum((mu_ra[:, 0]*weighted_pmra[stars_in][nb_stars] + mu_ra[:, 1]*weighted_pmdec[stars_in][nb_stars]))
        tau_mu_dec[i] = sum((mu_dec[:, 0]*weighted_pmra[stars_in][nb_stars] + mu_dec[:, 1]*weighted_pmdec[stars_in][nb_stars]))
        
        tau_mu_mon_norm[i] = np.sqrt(sum(mu_mon_sq/pm_w_sq[stars_in][nb_stars])) ## normalization
        tau_mu_mon[i] = sum((mu_mon[:, 0]*weighted_pmra[stars_in][nb_stars] + mu_mon[:, 1]*weighted_pmdec[stars_in][nb_stars]))
                    
    return np.array([scan_coord[0], scan_coord[1], tau_mu_ra, tau_mu_dec, tau_mu_norm, tau_mu_mon, tau_mu_mon_norm]).T



def fn_fine_scan_loc(sky_p, beta_t_deg, beta_t, list_tau_dir, frac, beta_step):
    """
    Finds the location of the largest tau values from the coarse scanning.
    Returns the location around tau values > frac*tau_max at beta_step of the coarse pixelation scale.
    """
    
    len_dfn = len(sky_p.data_file_name)

    ### List of files in the directory with the results from the coarse tau scan
    list_files = os.listdir(list_tau_dir)
    ### Select only files corresponding to the correct sky patch and beta_t value
    list_files = [file for file in list_files if (file[:len_dfn]==sky_p.data_file_name) & (file[len_dfn+6:len_dfn+7+len(beta_t_deg)]==beta_t_deg+'_')] 

    if len(list_files) < 1:
        print('ERROR: cannot find any file for the selected beta_t value of '+str(beta_t/degree)+' deg.')
        sys.stdout.flush()
        sys.exit_t
    print(str(len(list_files))+' files found for beta_t_deg value of '+beta_t_deg+' deg.')
    
    ### Find the largest tau value among all the values obtained from the coarse scanning    
    [ind_ra, ind_dec, ind_tau_ra, ind_tau_dec, ind_n, ind_tau_mon, ind_tau_mon_n] = range(7)
    tau_ra_max, tau_dec_max = 0, 0

    empty_file = 0
    for file in list_files:
        tau_values = np.load(list_tau_dir+file)
        if len(tau_values) > 0:
            tau_ra_max_temp, tau_dec_max_temp = np.max(np.abs(tau_values[:, ind_tau_ra])), np.max(np.abs(tau_values[:, ind_tau_dec]))
            if tau_ra_max_temp > tau_ra_max:
                tau_ra_max = tau_ra_max_temp
            if tau_dec_max_temp > tau_dec_max:
                tau_dec_max = tau_dec_max_temp
        else:
            empty_file=empty_file+1
    if empty_file > 0:
        print(empty_file, 'files were empty.')
        sys.stdout.flush()
                
    ### Find all the locations where tau > frac*tau_max. 
    large_tau = []
    for file in list_files:
        tau_values = np.load(list_tau_dir+file)

        large_tau_loc = np.array(np.where((np.abs(tau_values[:, ind_tau_ra]) > frac*tau_ra_max) | (np.abs(tau_values[:, ind_tau_dec]) > frac*tau_dec_max)))[0]
        large_tau.extend(tau_values[large_tau_loc])

    large_tau = np.array(large_tau)
    print(len(large_tau), 'large tau values.')
    
    ### For each large tau location, take positions at +-(1, 2)*beta_step*beta_pix_coarse (within a circle of radius beta_pix_coarse). 
    ### These are the locations where we will do the fine scanning.
    ### Coarse pixelation scale
    n_coarse = math.ceil(math.log(np.sqrt(np.pi/3)/beta_t, 2)); 
    beta_pix_coarse = np.sqrt(4*np.pi / hp.nside2npix(2**n_coarse))
    
    ### Locations around the large tau values where to do the fine scanning
    beta_steps = np.array([-2, -1, 0, 1, 2])*beta_step*beta_pix_coarse/degree
    fine_scan_ra, fine_scan_dec = [], []

    for i_loc in range(len(large_tau)):
        x_list = large_tau[i_loc, ind_ra] + beta_steps/np.cos(large_tau[i_loc, ind_dec]*degree)
        y_list = large_tau[i_loc, ind_dec] + beta_steps

        xy_grid = np.meshgrid(x_list, y_list, indexing='xy')
        x_grid_flat = xy_grid[0].flatten(); y_grid_flat = xy_grid[1].flatten()

        xy_dist = fn_angular_sep_scalar(large_tau[i_loc, ind_ra]*degree, large_tau[i_loc, ind_dec]*degree, x_grid_flat*degree, y_grid_flat*degree)

        fine_scan_ra.extend(x_grid_flat[(xy_dist < beta_pix_coarse) & (xy_dist > 0 )]); fine_scan_dec.extend(y_grid_flat[(xy_dist < beta_pix_coarse) & (xy_dist > 0 )])

    return [fine_scan_ra, fine_scan_dec], large_tau


