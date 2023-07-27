import numpy as np
import pandas as pd
import sys 

from utils.my_units import *
from utils.angular_fn import *


class Template():
    
    def __init__(self, template_type='proper motion', matched_filter='dipole', rot_angle=0, observation_t=2.833*Year):
        
        if template_type in ['proper motion', 'parallax']:        
            self.template_type = template_type
            if matched_filter in ['monopole', 'dipole', 'quadrupole']:
                self.matched_filter = matched_filter
            else:
                print('Error: matched filter = '+str(matched_filter)+' not supported.')            
                sys.exit()
        else:
            print('Error: template type = '+str(template_type)+' not supported.')
            sys.exit()
            
        self.rot_angle = rot_angle
        self.observation_t = observation_t

    
    def dipole_mf(self, vhat, bilhat):
        '''
        Function to compute unit dipole vector'''
        if vhat.shape == (2,):
            return np.array([vhat[0] - 2*bilhat[0]*(bilhat @ vhat) ,vhat[1] - 2*bilhat[1]*(bilhat @ vhat)])
        else:
            bildotv = bilhat[:,0]*vhat[:,0] + bilhat[:,1]*vhat[:,1]
            return np.array([vhat[:, 0] - 2*bilhat[:, 0]*bildotv, vhat[:, 1] - 2*bilhat[:, 1]*bildotv]).T
                
    def quadrupole_mf(self, vhat, bilhat):
        '''
        Function to compute unit quadrupole vector'''
        if vhat.shape == (2,):
            bildotv = (bilhat @ vhat)
            return np.array([2*vhat[0]*bildotv + bilhat[0]*(1-4*bildotv**2), 2*vhat[1]*bildotv + bilhat[1]*(1-4*bildotv**2)])
        else:
            bildotv = bilhat[:,0]*vhat[:,0] + bilhat[:,1]*vhat[:,1]
            return np.array([2*vhat[:,0]*bildotv + bilhat[:,0]*(1-4*bildotv**2), 2*vhat[:,1]*bildotv + bilhat[:,1]*(1-4*bildotv**2)]).T
        
        
    def lensing_pm(self, fg_ra, fg_dec, fg_pmra, fg_pmdec, fg_dist, bg_ra, bg_dec, bg_pmra, bg_pmdec, bg_dist):
        '''
        Function to compute the proper motion lensing correction for the list of foreground (lens) and background (source) pairs'''
        bil_vec = fn_angular_sep(fg_ra*degree, fg_dec*degree, bg_ra*degree, bg_dec*degree)
        bil_normsq = bil_vec[:, 0]**2 + bil_vec[:, 1]**2
        bil_hat = np.array([bil_vec[:, 0]/np.sqrt(bil_normsq), bil_vec[:, 1]/np.sqrt(bil_normsq)]).T 
        mu_vec = np.array([fg_pmra - bg_pmra, fg_pmdec - bg_pmdec]).T*mas/Year 
        mu_norm = np.sqrt(mu_vec[:, 0]**2 + mu_vec[:, 1]**2)
        mu_hat = np.array([mu_vec[:, 0]/mu_norm, mu_vec[:, 1]/mu_norm]).T 
        Dl = fg_dist*pc
        
        if self.matched_filter=='monopole':
            mf_vec = bil_hat
        elif self.matched_filter=='dipole':
            mf_vec = self.dipole_mf(mu_hat, bil_hat)
        elif self.matched_filter=='quadrupole':
            mf_vec = self.quadrupole_mf(mu_hat, bil_hat)
        else:
            print('Error: matched filter = '+str(self.matched_filte)+' not supported.')            
            sys.exit()
            
        lens_pm = np.array([-(1-fg_dist/bg_dist)*4*GN*MSolar*mu_norm/(fg_dist*pc)/bil_normsq*mf_vec[:, 0], 
                            -(1-fg_dist/bg_dist)*4*GN*MSolar*mu_norm/(fg_dist*pc)/bil_normsq*mf_vec[:, 1]]).T

        return lens_pm/(mas/Year), bil_normsq > (mu_norm*self.observation_t)**2

    
    def get_weights(self, ra_err, dec_err, radec_err_corr, no_corr=False):
        '''
        Function to compute the weights for the template'''
        if no_corr:
            cov_matrix = np.array([[ra_err**2, np.zeros(len(ra_err))],
                                   [np.zeros(len(ra_err)), dec_err**2]]).T
        else:
            cov_matrix = np.array([[ra_err**2, radec_err_corr*ra_err*dec_err],
                                   [radec_err_corr*ra_err*dec_err, dec_err**2]]).T
            
        return np.linalg.inv(cov_matrix)    
            
        
    def template_mu(self, df_fore, df_back, no_corr=False, mass_weight=None, return_list=False):
        '''
        Function to compute the proper motion template for the foreground (lens) and background (source) pairs'''
        
        lens_pm, template_cond = self.lensing_pm(df_fore['ra'].to_numpy(), df_fore['dec'].to_numpy(), 
                                                 df_fore['pmra'].to_numpy(),  df_fore['pmdec'].to_numpy(), df_fore['dist_50'].to_numpy(),
                                                 df_back['ra'].to_numpy(), df_back['dec'].to_numpy(),
                                                 df_back['pmra'].to_numpy(), df_back['pmdec'].to_numpy(), df_back['dist_50'].to_numpy())
        
        if self.rot_angle:
            print('Rotating the dipole profile by', self.rot_angle/degree, 'deg')
            costh, sinth = np.cos(self.rot_angle), np.sin(self.rot_angle)
            lens_pm = np.array([lens_pm[:,0]*costh-lens_pm[:,1]*sinth, lens_pm[:,1]*costh+lens_pm[:,0]*sinth]).T
    
        if mass_weight: 
            print('Including mass weights for the lenses: ', mass_weight)
            lens_pm = np.array([df_fore[mass_weight].to_numpy()*lens_pm[:,0], df_fore[mass_weight].to_numpy()*lens_pm[:,1]]).T
    
        # proper motion error weights
        weights = self.get_weights(df_back['pmra_eff_error'].to_numpy(), df_back['pmdec_eff_error'].to_numpy(), df_back['pmra_pmdec_eff_corr'].to_numpy(), no_corr)

        observed_pm = np.array([df_back['pmra_sub'].to_numpy(), df_back['pmdec_sub'].to_numpy()]).T
        norm_sq = weights[:, 0, 0]*lens_pm[:, 0]**2 + weights[:, 1, 1]*lens_pm[:, 1]**2 + 2*weights[:, 0, 1]*lens_pm[:, 0]*lens_pm[:, 1]
        tau_mu = (weights[:, 0, 0]*lens_pm[:, 0]*observed_pm[:, 0] + weights[:, 1, 1]*lens_pm[:, 1]*observed_pm[:, 1] + 
                  weights[:, 0, 1]*(lens_pm[:, 0]*observed_pm[:, 1] + lens_pm[:, 1]*observed_pm[:, 0]))

        if return_list: 
            # return the list of taus and normalizations for each stellar pair and the mask for pairs that satisfy the template condition
            return tau_mu, norm_sq, template_cond
        else:
            if(len(tau_mu[template_cond]) < len(tau_mu)):
                print((len(tau_mu)-len(tau_mu[template_cond])), ' stellar pairs do not satisfy the template condition (impact parameter is too small).')

            return sum(tau_mu[template_cond]), np.sqrt(sum(norm_sq[template_cond]))