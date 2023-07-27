import numpy as np
import math
import healpy as hp
import sys
from my_units import *
from sim_setup_fn import *
from sim_injection_fn import *
from template_fn import *

_sigma_vl = 166*1000*Meter/Second #_v0_ss/math.sqrt(2) ### Dark Matter velocity dispersion

def fn_chi_sq(M_l, r_l, beta_t, tau_values, sky_p):
    """
    Computes the chi_sq (optimal global test statistic) for each of the tau_values. The minimum should be retained.
    Returns an array of locations, beta_t values and the chi_sq values (with proper motion only).
    """

    [ind_ra, ind_dec, ind_tau_ra, ind_tau_dec, ind_n, ind_tau_mon, ind_tau_mon_n] = range(7)
    
    ### Converting to natural units
    n_mu_list = tau_values[:, ind_n]*(Year/mas) 
    t_mu_ra_list, t_mu_dec_list = tau_values[:, ind_tau_ra]*(Year/mas), tau_values[:, ind_tau_dec]*(Year/mas)
    t_mu_sq = t_mu_ra_list**2 + t_mu_dec_list**2

    v0 = -fn_obs_vel(tau_values[:, ind_ra]*degree, tau_values[:, ind_dec]*degree) # prefered direction for the velocity template
    v0_sq = v0[:, 0]**2 + v0[:, 1]**2
    tau_dot_v0 = t_mu_ra_list*v0[:, 0] + t_mu_dec_list*v0[:, 1] 
    
    C_l = 4*GN*M_l/r_l**2
    Cl_sigmav_Nmu = C_l*_sigma_vl*n_mu_list
    remove_small = np.heaviside(Cl_sigmav_Nmu-1, 0)
    
    t_mu_sq_over_n_sq = np.divide(t_mu_sq, n_mu_list**2, out=np.zeros_like(n_mu_list), where=n_mu_list!=0)
    tau_dot_over_n = np.divide(tau_dot_v0, n_mu_list*_sigma_vl, out=np.zeros_like(n_mu_list), where=n_mu_list!=0)
    mu_templ_contr = -0.5*(Cl_sigmav_Nmu)/(1+Cl_sigmav_Nmu**2)*( Cl_sigmav_Nmu*(t_mu_sq_over_n_sq - v0_sq/_sigma_vl**2) + 2*tau_dot_over_n )
           
    rho_beta_t_contr =  np.full((len(tau_values)), 
                                +4*math.log(beta_t) - math.log(fn_rho_dm(r_l/beta_t, sky_p.center_l*degree, sky_p.center_b*degree)/fn_rho_dm(sky_p.distance, sky_p.center_l*degree, sky_p.center_b*degree)) )
    
    return np.array([tau_values[:, ind_ra], tau_values[:, ind_dec], np.full((len(tau_values)), beta_t), (rho_beta_t_contr + mu_templ_contr)*remove_small ]).T


def fn_run_analysis(data, beta_t, n_betat, min_mask, beta_step, M_l, r_l, n_lens, lens_pop, sky_p):
    """
    Computes the templates for the given value of beta_t and the given list of lenses. Returns the list of chi^2 for each location.
    """
    ### Coarse pixelation scale   
    n_coarse = math.ceil(math.log(np.sqrt(np.pi/3)/beta_t, 2)); 
    beta_pix_coarse = np.sqrt(4*np.pi / hp.nside2npix(2**n_coarse))
    
    ### Take 5 locations around the lens (ra, dec) where to compute the template using the step of the fine pixelation
    ### Taking 20 locations per lens
    beta_steps = np.array([-2, -1, 0, 1, 2])*beta_step*beta_pix_coarse/degree
    template_scan_ra, template_scan_dec = [], []
    
    if n_lens==1:
        x_list = lens_pop[0] + beta_steps/np.cos(lens_pop[1]*degree)
        y_list = lens_pop[1] + beta_steps

        xy_grid = np.meshgrid(x_list, y_list, indexing='xy')
        x_grid_flat = xy_grid[0].flatten(); y_grid_flat = xy_grid[1].flatten()

        xy_dist = fn_angular_sep_scalar(lens_pop[0]*degree, lens_pop[1]*degree, x_grid_flat*degree, y_grid_flat*degree)
        template_scan_ra.extend(x_grid_flat[xy_dist < beta_pix_coarse]); template_scan_dec.extend(y_grid_flat[xy_dist < beta_pix_coarse])
        
    else:
        for i in range(len(lens_pop)):
            x_list = lens_pop[i, 0] + beta_steps/np.cos(lens_pop[i, 1]*degree)
            y_list = lens_pop[i, 1] + beta_steps

            xy_grid = np.meshgrid(x_list, y_list, indexing='xy')
            x_grid_flat = xy_grid[0].flatten(); y_grid_flat = xy_grid[1].flatten()

            xy_dist = fn_angular_sep_scalar(lens_pop[i, 0]*degree, lens_pop[i, 1]*degree, x_grid_flat*degree, y_grid_flat*degree)
            template_scan_ra.extend(x_grid_flat[xy_dist < beta_pix_coarse]); template_scan_dec.extend(y_grid_flat[xy_dist < beta_pix_coarse])
        
    ### Fine pixelation of size approx. beta_t/10 (can be a bit larger than beta_t/10, so using round is fine)
    n = round(math.log(np.sqrt(np.pi/3)/(0.1*beta_t), 2)); nside = 2**n; 
    template_scan_pix = np.unique(hp.ang2pix(nside, template_scan_ra, template_scan_dec, nest=True, lonlat=True)) 
    n_locations = len(template_scan_pix)

    print('\nTemplate scan for beta_t = '+str(beta_t/degree)+' deg.')
    print('Number of template locations: '+str(n_locations))
    sys.stdout.flush()
    
    data_pix = np.asarray(hp.ang2pix(nside, data[:, 0], data[:, 1], nest=True, lonlat=True)) # healpy pixel number of the stars, needed to find stars near a specific template location
    
    ### Compute the template 
    tau_values = fn_template_scan(nside, template_scan_pix, n_betat, beta_t, np.array([data_pix] + [data[:, i] for i in range(len(data[0]))]).T, min_mask)
    
    print('Template scan completed. Computing the chi_sq...')
    sys.stdout.flush()
    chi_sq = fn_chi_sq(M_l, r_l, beta_t, tau_values, sky_p)
    
    return chi_sq

