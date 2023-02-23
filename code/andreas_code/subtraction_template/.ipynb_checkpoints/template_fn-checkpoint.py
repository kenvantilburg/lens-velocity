import numpy as np
import pandas as pd
from angular_fn import *
from my_units import *


### Functions for the matched-filters used to compute the templates

def fn_monopole_mf(vhat, bilhat):
    """
    Returns the unit monopole vector
    """
    return bilhat

def fn_dipole_mf(vhat, bilhat):
    """
    Returns the unit dipole vector
    """
    if vhat.shape == (2,):
        return np.array([vhat[0] - 2*bilhat[0]*(bilhat @ vhat) ,vhat[1] - 2*bilhat[1]*(bilhat @ vhat)])
    else:
        bildotv = bilhat[:,0]*vhat[:,0] + bilhat[:,1]*vhat[:,1]
        return np.array([vhat[:, 0] - 2*bilhat[:, 0]*bildotv, vhat[:, 1] - 2*bilhat[:, 1]*bildotv]).T
    
def fn_quadrupole_mf(vhat, bilhat):
    """
    Returns the unit quadrupole vector
    """
    if vhat.shape == (2,):
        bildotv = (bilhat @ vhat)
        return np.array([2*vhat[0]*bildotv + bilhat[0]*(1-4*bildotv**2), 2*vhat[1]*bildotv + bilhat[1]*(1-4*bildotv**2)])
    else:
        bildotv = bilhat[:,0]*vhat[:,0] + bilhat[:,1]*vhat[:,1]
        return np.array([2*vhat[:,0]*bildotv + bilhat[:,0]*(1-4*bildotv**2), 2*vhat[:,1]*bildotv + bilhat[:,1]*(1-4*bildotv**2)]).T
    
def fn_parallax_mf(bilhat, s_delta_sq):
    """
    Returns the lens-induced parallax profile
    """
    if bilhat.shape == (2,):
        return (2*bilhat[1]**2-1)*(1-s_delta_sq)/(1+s_delta_sq)
    else:
        return (2*bilhat[:,1]**2-1)*(1-s_delta_sq)/(1+s_delta_sq)
    
    
### Functions for the lens-induced proper motion, parallax and acceleration.

### !!!!! NOTICE that the functions below take into account the finite distance of the background source as well (in the magnitude of the lensing and in the relative velocity between the source and the lens). For background stars that have a negative parallax measurement and or compatible with zero at 2-3*sigma, the coefficient (1-bg_parallax/fg_parallax) should probably be discrded.

def fn_lensing_pm(fg_ra, fg_dec, fg_pmra, fg_pmdec, fg_parallax, bg_ra, bg_dec, bg_pmra, bg_pmdec, bg_parallax, tau_obs, matched_filter=fn_dipole_mf):
    """
    Computes the lens-induced proper motion on background stars due to the foreground stars. In the template regime, the impact parameter must be larger than the distance travelled by the lens during the observation time tau_obs.    
    Output: list of 2d proper motion vectors (in mas/y) and indices of pairs with impact parameter in the template regime (the other pairs should be discarded).
    """
    l_bilvec = fn_angular_sep(fg_ra*degree, fg_dec*degree, bg_ra*degree, bg_dec*degree)
    l_bilnormsq = l_bilvec[:, 0]**2 + l_bilvec[:, 1]**2
    l_bilhat = np.array([l_bilvec[:, 0]/np.sqrt(l_bilnormsq), l_bilvec[:, 1]/np.sqrt(l_bilnormsq)]).T 
    l_muvec = np.array([fg_pmra - bg_pmra, fg_pmdec - bg_pmdec]).T*mas/Year #np.array([fg_pmra, fg_pmdec]).T*mas/Year
    l_munorm = np.sqrt(l_muvec[:, 0]**2 + l_muvec[:, 1]**2)
    l_muhat = np.array([l_muvec[:, 0]/l_munorm, l_muvec[:, 1]/l_munorm]).T 
    l_Dl = 1/fg_parallax*kpc

    ### Evaluating the matched filter
    l_dipole_vec = matched_filter(l_muhat, l_bilhat)

    pm_list = np.array([-(1-bg_parallax/fg_parallax)*4*GN*MSolar*l_munorm/l_Dl/l_bilnormsq*l_dipole_vec[:, 0], 
                        -(1-bg_parallax/fg_parallax)*4*GN*MSolar*l_munorm/l_Dl/l_bilnormsq*l_dipole_vec[:, 1]]).T

    return pm_list[l_bilnormsq > (l_munorm*tau_obs)**2]/(mas/Year), np.arange(len(fg_ra))[l_bilnormsq > (l_munorm*tau_obs)**2]


def fn_lensing_acc(fg_ra, fg_dec, fg_pmra, fg_pmdec, fg_parallax, bg_ra, bg_dec, bg_pmra, bg_pmdec, bg_parallax, tau_obs, matched_filter=fn_quadrupole_mf):
    """
    Computes the lens-induced angular acceleration on background stars due to the foreground stars. In the template regime, the impact parameter must be larger than the distance travelled by the lens during the observation time tau_obs.   
    Output: list of 2d acceleration vectors (in mas/y^2) and indices of pairs with impact parameter in the template regime (the other pairs should be discarded).
    """
    l_bilvec = fn_angular_sep(fg_ra*degree, fg_dec*degree, bg_ra*degree, bg_dec*degree)
    l_bilnormsq = l_bilvec[:, 0]**2 + l_bilvec[:, 1]**2
    l_bilhat = np.array([l_bilvec[:, 0]/np.sqrt(l_bilnormsq), l_bilvec[:, 1]/np.sqrt(l_bilnormsq)]).T 
    l_muvec = np.array([fg_pmra - bg_pmra, fg_pmdec - bg_pmdec]).T*mas/Year #np.array([fg_pmra, fg_pmdec]).T*mas/Year
    l_munormsq = l_muvec[:, 0]**2 + l_muvec[:, 1]**2
    l_muhat = np.array([l_muvec[:, 0]/np.sqrt(l_munormsq), l_muvec[:, 1]/np.sqrt(l_munormsq)]).T 
    l_Dl = 1/fg_parallax*kpc

    """Evaluating the matched filter"""
    l_quad_vec = matched_filter(l_muhat, l_bilhat)

    acc_list = np.array([(1-bg_parallax/fg_parallax)*8*GN*MSolar*l_munormsq/l_Dl/(l_bilnormsq*np.sqrt(l_bilnormsq))*l_quad_vec[:, 0], 
                         (1-bg_parallax/fg_parallax)*8*GN*MSolar*l_munormsq/l_Dl/(l_bilnormsq*np.sqrt(l_bilnormsq))*l_quad_vec[:, 1]]).T

    return acc_list[l_bilnormsq > l_munormsq*tau_obs**2]/(mas/Year**2), np.arange(len(fg_ra))[l_bilnormsq > l_munormsq*tau_obs**2]


def fn_lensing_prallax(fg_ra_ecl, fg_dec_ecl, fg_parallax, bg_ra_ecl, bg_dec_ecl, bg_parallax, matched_filter=fn_parallax_mf):
    """
    Computes the lens-induced parallax on background stars due to the foreground stars. In the template regime, the impact parameter must be larger than the foreground star's parallax.
    Output: list of parallaxes (in mas) and indices of pairs with impact parameter in the template regime (the other pairs should be discarded).
    """
    l_bilvec = fn_angular_sep(fg_ra_ecl*degree, fg_dec_ecl*degree, bg_ra_ecl*degree, bg_dec_ecl*degree)    
    l_bilnormsq = l_bilvec[:, 0]**2 + l_bilvec[:, 1]**2
    l_bilhat = np.array([l_bilvec[:, 0]/np.sqrt(l_bilnormsq), l_bilvec[:, 1]/np.sqrt(l_bilnormsq)]).T 
    l_Dl = 1/fg_parallax*kpc
    s_delta_sq = np.sin(fg_dec_ecl*degree)**2
        
    parallax_list = -(1-bg_parallax/fg_parallax)*4*GN*MSolar/l_Dl*fg_parallax/l_bilnormsq*matched_filter(l_bilhat, s_delta_sq)
      
    return parallax_list[np.sqrt(l_bilnormsq)/mas > fg_parallax], np.arange(len(fg_ra_ecl))[np.sqrt(l_bilnormsq)/mas > fg_parallax]



### Functions to compute the templates.


def fn_tau_mu(df_fore, df_back, weights, tau_obs, tau_max=False, matched_filter=fn_dipole_mf, rot_angle=False, quiet=True):
    """
    For a given data frame of foreground and background stars, return the proper motion tau test statistic and its normalization.
    Weights must be an array of 2x2 matrices of the same lenght of df_fore and df_back, e.g. the pm inverse covariance matrix for each background star.
    If tau_max!=False, only the values of tau < tau_max are kept in the sum over all pairs. 
    """
    
    if not quiet: print('Computing the expected velocity.')
    lensing_pm, good_bil_ind = fn_lensing_pm(df_fore['ra'].to_numpy(), df_fore['dec'].to_numpy(),
                                             df_fore['pmra'].to_numpy(),  df_fore['pmdec'].to_numpy(), df_fore['parallax'].to_numpy(),
                                             df_back['ra'].to_numpy(), df_back['dec'].to_numpy(),
                                             df_back['pmra'].to_numpy(), df_back['pmdec'].to_numpy(), df_back['parallax'].to_numpy(), tau_obs, matched_filter)

    if(len(lensing_pm) < len(df_back)):
        if not quiet: print((len(df_back)-len(lensing_pm)), 'star pairs have too small impact parameter. Selecting only the good pairs.')
        df_back = df_back.iloc[good_bil_ind]; weights = weights[good_bil_ind]
    
    if rot_angle:
        if not quiet: print('Rotating the dipole profile by', rot_angle/degree, 'deg')
        costh, sinth = np.cos(rot_angle), np.sin(rot_angle)
        lensing_pm = np.array([lensing_pm[:,0]*costh-lensing_pm[:,1]*sinth, lensing_pm[:,1]*costh+lensing_pm[:,0]*sinth]).T


    if not quiet: print('Computing the tau lists.')
    ### Observed background stars' proper motion (after subtraction)
    observed_pm = np.array([df_back['pmra_sub'].to_numpy(), df_back['pmdec_sub'].to_numpy()]).T
    tau_norm_sq = weights[:, 0, 0]*lensing_pm[:, 0]**2 + weights[:, 1, 1]*lensing_pm[:, 1]**2 + 2*weights[:, 0, 1]*lensing_pm[:, 0]*lensing_pm[:, 1]
    tau_mu = (weights[:, 0, 0]*lensing_pm[:, 0]*observed_pm[:, 0] + weights[:, 1, 1]*lensing_pm[:, 1]*observed_pm[:, 1] + 
              weights[:, 0, 1]*(lensing_pm[:, 0]*observed_pm[:, 1] + lensing_pm[:, 1]*observed_pm[:, 0]))

    if not quiet: print('Computing the sum.')
    if tau_max!=0:
        small_tau_mu = tau_mu[np.abs(tau_mu) < tau_max]
        if not quiet: print(str((1-len(small_tau_mu)/len(tau_mu))*100)+'% of taus removed')
        return sum(small_tau_mu), np.sqrt(sum(tau_norm_sq[np.abs(tau_mu) < tau_max]))
    else:
        return sum(tau_mu), np.sqrt(sum(tau_norm_sq))
    

    
def fn_tau_acc(df_fore, df_back, weights, tau_obs, tau_max=False, matched_filter=fn_quadrupole_mf, rot_angle=False, quiet=True):
    """
    For a given data frame of foreground and background stars, returns the acceleration tau test statistic and its normalization.
    Weights must be an array of 2x2 matrices of the same lenght of df_fore and df_back, e.g. the acceleration inverse covariance matrix for each background star.
    If tau_max!=False, only the values of tau < tau_max are kept in the sum over all pairs. 
    """
    lensing_acc, good_bil_ind = fn_lensing_acc(df_fore['ra'].to_numpy(), df_fore['dec'].to_numpy(),
                                               df_fore['pmra'].to_numpy(),  df_fore['pmdec'].to_numpy(), df_fore['parallax'].to_numpy(), 
                                               df_back['ra'].to_numpy(), df_back['dec'].to_numpy(), 
                                               df_back['pmra'].to_numpy(),  df_back['pmdec'].to_numpy(), df_back['parallax'].to_numpy(), tau_obs, matched_filter)

    if(len(exp_acc) < len(df_fore)):
        if not quiet: print((len(df_fore)-len(lensing_acc)), 'star pairs have too small impact parameter. Selecting only the good pairs.')
        df_back = df_back.iloc[good_bil_ind]; weights = weights[good_bil_ind];
        
    if rot_angle:
        if not quiet: print('Rotating the dipole profile by', rot_angle/degree, 'deg')
        costh, sinth = np.cos(rot_angle), np.sin(rot_angle)
        lensing_acc = np.array([lensing_acc[:,0]*costh-lensing_acc[:,1]*sinth, lensing_acc[:,1]*costh+lensing_acc[:,0]*sinth]).T


    if not quiet: print('Computing the tau lists.')
    ### Observed background stars' acceleration (after subtraction)
    observed_acc = np.array([df_back['accra_sub'].to_numpy(), df_back['accdec_sub'].to_numpy()]).T
    tau_norm_sq = weights[:, 0, 0]*lensing_acc[:, 0]**2 + weights[:, 1, 1]*lensing_acc[:, 1]**2 + 2*weights[:, 0, 1]*lensing_acc[:, 0]*lensing_acc[:, 1]
    tau_acc = (weights[:, 0, 0]*lensing_acc[:, 0]*observed_acc[:, 0] + weights[:, 1, 1]*lensing_acc[:, 1]*observed_acc[:, 1] + 
               weights[:, 0, 1]*(lensing_acc[:, 0]*observed_acc[:, 1] + lensing_acc[:, 1]*observed_acc[:, 0]))   
    
    if not quiet: print('Computing the sum.')
    if tau_max!=0:
        small_tau_acc = tau_acc[np.abs(tau_acc) < tau_max]
        return sum(small_tau_acc), np.sqrt(sum(tau_norm_sq[np.abs(tau_acc) < tau_max]))
    else:
        return sum(tau_acc), np.sqrt(sum(tau_norm_sq))
    
    

def fn_varpi(df_fore, df_back, weights, tau_max=False, matched_filter=fn_parallax_mf, quiet=True):
    """
    For a given data frame of foreground and background stars, returns the parallax test statistic tau and its normalization.
    Weights must be an array of the same lenght of df_fore and df_back, e.g. the parallax inverse variance of each background star.
    If tau_max!=False, only the values of tau < tau_max are kept in the sum over all pairs. 
    """
    lensing_varpi, good_bil_ind = fn_lensing_prallax(df_fore['ecl_lon'].to_numpy(), df_fore['ecl_lat'].to_numpy(), df_fore['parallax'].to_numpy(),
                                                     df_back['ecl_lon'].to_numpy(), df_back['ecl_lat'].to_numpy(), df_back['parallax'].to_numpy(), matched_filter)

    if(len(lensing_varpi) < len(df_back)):
        if not quiet: print((len(df_back)-len(lensing_varpi)), 'star pairs have too small impact parameter. Selecting only the good pairs.')
        df_back = df_back.iloc[good_bil_ind]
    
    ### Observed parallaxes 
    varpi_observed = df_back['parallax_sub'].to_numpy() 
    
    tau_norm_sq = lensing_varpi**2*weights
    tau_varpi = lensing_varpi*varpi_observed*weights
       
    if tau_max!=0:
        small_tau_varpi = tau_varpi[np.abs(tau_varpi) < tau_max]
        return sum(small_tau_varpi), np.sqrt(sum(tau_norm_sq[np.abs(tau_varpi) < tau_max]))
    else:
        return sum(tau_varpi), np.sqrt(sum(tau_norm_sq))    