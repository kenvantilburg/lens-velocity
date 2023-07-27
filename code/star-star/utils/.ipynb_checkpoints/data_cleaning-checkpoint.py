import numpy as np
import pandas as pd
import scipy as sp

from utils.my_units import *
from utils.angular_fn import *


class CleanData():
    
    def __init__(self, ruwe_cut=1.4, n_sigma=2):
        self.ruwe_cut = ruwe_cut
        self.n_sigma = n_sigma
    
    def setup_params(self, default=True, ruwe_cut=1.4, n_sigma=2):
        '''
        Function to set up the parameters for cleaning'''
        if default:
            self.ruwe_cut = 1.4
            self.n_sigma = 2
        else:
            self.ruwe_cut = ruwe_cut
            self.n_sigma = n_sigma    
        try:
            1/self.ruwe_cut
            1/self.n_sigma
        except:
            print('Error with the initialized parameters.')    
            
    def astrometric_quality(self, df_fore, df_back):
        '''
        Function to remove all foreground-background pairs in which either one of the stars in the pair doesn't have a good astrometric solution'''
        
        good_rows = ( (df_fore['ruwe'] < self.ruwe_cut) & (df_back['ruwe'] < self.ruwe_cut) ) 
        
        len_before = len(df_fore)
        rem_frac = (1-len(df_fore[good_rows])/len_bg)
        print('Fraction of stellar pairs with poor astrometric solution: '+str(rem_frac*100)[:8]+' %')
        
        return df_fore.iloc[good_rows], df_back.iloc[good_rows]        
        
    def distance_cut(self, df_fore, df_back):
        '''
        Function to remove stellar pairs whose difference between distances is less than self.n_sigma apart'''
        
        sigma_86_fore = df_fore['dist_86'].to_numpy()-df_fore['dist_50'].to_numpy()
        sigma_14_back = df_back['dist_50'].to_numpy()-df_back['dist_14'].to_numpy()
        
        good_rows = (df_back['dist_50'].to_numpy() - df_fore['dist_50'].to_numpy()) > self.n_sigma*np.sqrt( sigma_86_fore**2 + sigma_14_back**2 )
        
        len_before = len(df_fore)
        rem_frac = (1-len(df_fore[good_rows])/len_bg)
        print('Fraction of stellar pairs with bad distances: '+str(rem_frac*100)[:8]+' %')
        
        return df_fore.iloc[good_rows], df_back.iloc[good_rows]            

    
    

class ProperMotionStats():

    def __init__(self, HEALpix_n=0, bins_dist=0, bins_G=0, bins_bil=0, th_count=30, prob_sparse_th = 0.2, n_sigma_out = 3):
        self.HEALpix_n = HEALpix_n
        self.bins_dist = bins_dist
        self.bins_G = bins_G
        self.bins_bil = bins_bil
        
        self.th_count=th_count, 
        self.prob_sparse_th = prob_sparse_th, 
        self.n_sigma_out = n_sigma_out

    def setup_bins(self, default=True, HEALpix_n=0, bins_dist=0, bins_G=0, bins_bil=0):
        '''
        Function to sut up the bins for the 4d histogram'''
        if default:
            self.HEALpix_n = 4                                                                                  # nested HEALPix scheme level
            self.bins_dist = np.concatenate([[0], np.logspace(np.log10(1000), np.log10(10000), 5), [200000]])   # distance bins
            self.bins_G = np.arange(5,23,1)                                                                     # G magnitude bins
            self.bins_bil = np.arange(0, 3.3, 0.3)                                                              # angular separation (impact parameter) bins
        else:
            self.HEALpix_n = HEALpix_n                                                                                 
            self.bins_dist = bins_dist   
            self.bins_G = bins_G                                                                  
            self.bins_bil = bins_bil                                                             
            
        try:
            1/self.HEALpix_n
            len(self.bins_dist)
            len(self.bins_G)
            len(self.bins_bil)
        except:
            print('Error with the bins definition.')    
            
    def setup_params(self, default=True, th_count=30, prob_sparse_th = 0.2, n_sigma_out = 3):
        '''
        Function to set up the parameters including the threshold count for sparse bins, the threshold probability support in sparse bins, and the number of sigmas for removing outliers'''
        if default:
            self.th_count = 30
            self.prob_sparse_th = 0.2
            self.n_sigma_out = 3
        else:
            self.th_count=th_count, 
            self.prob_sparse_th = prob_sparse_th, 
            self.n_sigma_out = n_sigma_out
            
        try:
            1/self.th_count
            1/self.prob_sparse_th
            1/self.n_sigma_out
        except:
            print('Error with the initialized parameters.')    
            

    def bin_assigment(self, df_fore, df_back):
        '''
        Function to assign the background sources from the data frame df_back to the bins of the 4d histogram'''
        
        # assign to healpix bins
        fac_source_id = 2**(59-2*self.HEALpix_n) # factorization used to extract the healpy bin from the source id
        q_pix = np.floor(df_back['source_id'].to_numpy() / fac_source_id).astype(int)
        bins_pix = np.arange(np.min(np.unique(q_pix)), np.max(np.unique(q_pix))+2,1) # should be +2 to include sources in the last bin
        q_bin_pix = np.digitize(q_pix, bins_pix) - 1  # need to access the histogram matrix elements

        # assign to G bins
        bg_G = df_back['phot_g_mean_mag'].to_numpy()
        q_bin_G = np.digitize(bg_G, self.bins_G) - 1 
        
        # assign to radial bins
        l_bilvec = fn_angular_sep(df_fore['ra'].to_numpy()*degree, df_fore['dec'].to_numpy()*degree, 
                                  df_back['ra'].to_numpy()*degree, df_back['dec'].to_numpy()*degree) # bil separation vectors
        q_bil = np.sqrt(l_bilvec[:, 0]**2 + l_bilvec[:, 1]**2)/arcsec
        q_bin_bil = np.digitize(q_bil, self.bins_bil) - 1   
        
        # probabilistic assignment to distance bins
        bg_dist = df_back['dist_50'].to_numpy()
        bg_dist_error = df_back['dist_error'].to_numpy()
        
        bins_dist_neg = np.concatenate([[-100000], self.bins_dist]) # including a nagative distance bin for the stars that overflow
        prob_dist_neg = np.nan * np.ones((len(df_back),len(bins_dist_neg)-1))
        
        for i in range(len(bins_dist_neg)-1):
            x1_list = (bins_dist_neg[i]-bg_dist)/bg_dist_error/np.sqrt(2)
            x2_list = (bins_dist_neg[i+1]-bg_dist)/bg_dist_error/np.sqrt(2)
            prob_dist_neg[:,i] = 0.5*(sp.special.erf(x2_list)-sp.special.erf(x1_list))

        # add the probability in the negative distance bin to the first non negative bin
        prob_dist = np.nan * np.ones((len(df_back),len(self.bins_dist)-1))
        prob_dist = np.copy(prob_dist_neg[:, 1:])
        prob_dist[:, 0] += prob_dist_neg[:, 0]
        
        return q_pix, bins_pix, q_bin_pix, bg_G, q_bin_G, q_bil, q_bin_bil, prob_dist
    
    
    def fill_histo(self, bg_pmra, bg_pmdec, q_pix, bins_pix, q_bin_pix, bg_G, q_bin_G, q_bil, q_bin_bil, prob_dist):
        '''
        Function to fill in the 4d histogram using the background stars pm_ra, pm_dec and parallax'''
        
        # histogram of summed probabilities
        hist_prob = sp.stats.binned_statistic_dd([q_bil, bg_G, q_pix], np.transpose(prob_dist), bins=[self.bins_bil, self.bins_G, bins_pix],statistic='sum')[0] 
        # histogram of average pmra weighted by probabilities
        hist_pmra = sp.stats.binned_statistic_dd([q_bil, bg_G, q_pix],np.transpose(prob_dist) * bg_pmra, bins=[self.bins_bil, self.bins_G, bins_pix],statistic='sum')[0] #sum first in each bin
        hist_pmra = hist_pmra / (hist_prob + 1e-20) #then divide by number in each bin
        hist_pmdec = sp.stats.binned_statistic_dd([q_bil, bg_G, q_pix],np.transpose(prob_dist) * bg_pmdec, bins=[self.bins_bil, self.bins_G, bins_pix], statistic='sum')[0] #sum first in each bin
        hist_pmdec = hist_pmdec / (hist_prob + 1e-20) #then divide by number in each bin

        # for each star, get the mean pm and parallax of the corresponding bin
        mean_pmra = hist_pmra[:, q_bin_bil, q_bin_G, q_bin_pix].T; mean_pmdec = hist_pmdec[:, q_bin_bil, q_bin_G, q_bin_pix].T

        # histogram of pm variance weighted by dist bin probabilities
        hist_pmra_var = sp.stats.binned_statistic_dd([q_bil, bg_G, q_pix],np.transpose(prob_dist) * (mean_pmra.T - bg_pmra)**2,
                                                        bins=[self.bins_bil, self.bins_G, bins_pix],statistic='sum')[0] #sum first in each bin
        hist_pmra_var = hist_pmra_var / (hist_prob - 1 + 1e-20) # the estimator should have a -1 (this matches for example var() computed with panda's groupy)
        hist_pmdec_var = sp.stats.binned_statistic_dd([q_bil, bg_G, q_pix],np.transpose(prob_dist) * (mean_pmdec.T - bg_pmdec)**2,
                                                        bins=[self.bins_bil, self.bins_G, bins_pix],statistic='sum')[0] #sum first in each bin
        hist_pmdec_var = hist_pmdec_var / (hist_prob - 1 + 1e-20) 
        hist_pmradec_var = sp.stats.binned_statistic_dd([q_bil, bg_G, q_pix],np.transpose(prob_dist) * (mean_pmra.T - bg_pmra) * (mean_pmdec.T - bg_pmdec),
                                                            bins=[self.bins_bil, self.bins_G, bins_pix],statistic='sum')[0] #sum first in each bin
        hist_pmradec_var = hist_pmradec_var / (hist_prob - 1 + 1e-20) 

        return hist_prob, hist_pmra, hist_pmdec, hist_pmra_var, hist_pmdec_var, hist_pmradec_var    
    
    
    def get_stars_stats(self, hist_prob, hist_pmra, hist_pmdec, hist_pmra_var, hist_pmdec_var, hist_pmradec_var, q_bin_bil, q_bin_G, q_bin_pix, prob_dist):
        '''
        Function to obtain the mean and variance for proper motion and parallax of each background star'''
    
        # set to nan bins where there are too few stars
        hist_pmra[hist_prob < self.th_count] = np.nan; hist_pmdec[hist_prob < self.th_count] = np.nan; 
        hist_pmra_var[hist_prob < self.th_count] = np.nan; hist_pmdec_var[hist_prob < self.th_count] = np.nan; hist_pmradec_var[hist_prob < self.th_count] = np.nan

        # for each star, get the pm mean and variance of the corresponding bin (after excluding the low count bins)
        mean_pmra = hist_pmra[:, q_bin_bil, q_bin_G, q_bin_pix].T; mean_pmdec = hist_pmdec[:, q_bin_bil, q_bin_G, q_bin_pix].T
        var_pmra = hist_pmra_var[:, q_bin_bil, q_bin_G, q_bin_pix].T; var_pmdec = hist_pmdec_var[:, q_bin_bil, q_bin_G, q_bin_pix].T; var_pmradec = hist_pmradec_var[:, q_bin_bil, q_bin_G, q_bin_pix].T;    

        #  fet the mean and var for each star
        tab_sum_pw = np.sum(prob_dist, axis=1, where=(~np.isnan(mean_pmra)))  # sum of the dist weights for each star using only bins with enough statistics 
        tab_mean_pmra = np.sum(np.nan_to_num(mean_pmra*prob_dist), axis=1)/(tab_sum_pw + 1e-20)
        tab_mean_pmdec = np.sum(np.nan_to_num(mean_pmdec*prob_dist), axis=1)/(tab_sum_pw + 1e-20)
        tab_var_pmra = np.sum(np.nan_to_num(var_pmra*prob_dist), axis=1)/(tab_sum_pw + 1e-20)
        tab_var_pmdec = np.sum(np.nan_to_num(var_pmdec*prob_dist), axis=1)/(tab_sum_pw + 1e-20)
        tab_var_pmradec = np.sum(np.nan_to_num(var_pmradec*prob_dist), axis=1)/(tab_sum_pw + 1e-20)   

        return tab_mean_pmra, tab_mean_pmdec, tab_var_pmra, tab_var_pmdec, tab_var_pmradec
    
    
    def remove_sparse_prob(self, hist_prob_copy, q_bin_bil, q_bin_G, q_bin_pix, prob_dist):
        '''
        Function to remove stars with more than 20% probability support (self.prob_sparse_th) in sparse bins.'''
        
        hist_prob_copy = (hist_prob_copy < self.th_count).astype(int)
        count_prob_sparse = hist_prob_copy[:, q_bin_bil, q_bin_G, q_bin_pix].T

        prob_in_sparse_bin = np.sum(count_prob_sparse*prob_dist, axis=1)
        
        return prob_in_sparse_bin < self.prob_sparse_th                
        
    
    def compute_stats(self, df_fore, df_back, iter_n, final_call=False):
        '''
        Function that takes the raw data (df_fore, df_back) and computes the mean and variance of proper motion and parallax binning the stars in 4d histograms of (distance, b_il, G mag, HEALpix)'''
        
        q_pix, bins_pix, q_bin_pix, bg_G, q_bin_G, q_bil, q_bin_bil, prob_dist = self.bin_assigment(df_fore, df_back)
        
        hist_prob, hist_pmra, hist_pmdec, hist_pmra_var, hist_pmdec_var, hist_pmradec_var = self.fill_histo(df_back['pmra'].to_numpy(), df_back['pmdec'].to_numpy(), 
                                                                                                            q_pix, bins_pix, q_bin_pix, bg_G, q_bin_G, q_bil, q_bin_bil, prob_dist)
        
        tab_mean_pmra, tab_mean_pmdec, tab_var_pmra, tab_var_pmdec, tab_var_pmradec = self.get_stars_stats(hist_prob, hist_pmra, hist_pmdec, hist_pmra_var, hist_pmdec_var, hist_pmradec_var,
                                                                                                           q_bin_bil, q_bin_G, q_bin_pix, prob_dist)
        
        # find all the stars whose distance probability support is mostly in spare bins. Remove stars with more than 20% probability in sparse bins
        good_bins = self.remove_sparse_prob(np.copy(hist_prob), q_bin_bil, q_bin_G, q_bin_pix, prob_dist)

        len_bg = len(df_back)
        out_frac=(1-len(tab_mean_pmra[good_bins])/len_bg);        
        print('Iter '+str(iter_n)+' -- fraction of stars in sparse bins: '+str(out_frac*100)[:8]+' %')

        tab_mean_pmra = tab_mean_pmra[good_bins]; tab_mean_pmdec = tab_mean_pmdec[good_bins]; 
        tab_var_pmra = tab_var_pmra[good_bins]; tab_var_pmdec = tab_var_pmdec[good_bins]; tab_var_pmradec = tab_var_pmradec[good_bins]

        df_fore = df_fore.iloc[good_bins]; df_back = df_back.iloc[good_bins]
        
        # subtracted pm and inverse covariance for outlier removal
        pm_sub = np.array([df_back['pmra'].to_numpy()-tab_mean_pmra, df_back['pmdec'].to_numpy()-tab_mean_pmdec]).T
        
        if final_call==False: # remove outliers
            inv_cov_pm = np.linalg.inv(np.array([[tab_var_pmra, tab_var_pmradec], [tab_var_pmradec, tab_var_pmdec]]).T)
            mu_over_sigma_sq = inv_cov_pm[:, 0, 0]*pm_sub[:, 0]**2 + inv_cov_pm[:, 1, 1]*pm_sub[:, 1]**2 + 2*inv_cov_pm[:, 0, 1]*pm_sub[:, 0]*pm_sub[:, 1]

            not_outlier = (mu_over_sigma_sq < self.n_sigma_out**2)

            len_bg = len(df_back)
            df_fore = df_fore.iloc[not_outlier]
            df_back = df_back.iloc[not_outlier]
            out_frac=(1-len(df_back)/len_bg);     

            print('Iter '+str(iter_n)+' -- fraction of outliers removed: '+str(out_frac*100)[:8]+' %')
            
            return df_fore, df_back, out_frac
            
        else: # add columns corresponding to subtracted pm motion and effective variance 
            print('Adding columns with the new stats...')
                        
            df_back['pmra_sub'] = pm_sub[:, 0]; df_back['pmdec_sub_'] = pm_sub[:, 1]; 

            df_back['pmra_eff_error'] = np.sqrt(tab_var_pmra)
            df_back['pmdec_eff_error'] = np.sqrt(tab_var_pmdec)
            df_back['pmra_pmdec_eff_corr'] = tab_var_pmradec/(np.sqrt(tab_var_pmra)*np.sqrt(tab_var_pmdec))
        
            return df_fore, df_back