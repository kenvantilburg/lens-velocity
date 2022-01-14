import numpy as np
import healpy as hp
import math
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import scipy
from my_units import *
from angular_fn import *
from template_fn import *
from sim_setup_fn import *

def fn_noise_inj(data, sky_p, gmag_bin_size=0.1, rad_bin_size=1, noise=True):
    """
    Injects the data-driven noise in the stars proper motion. Add 2 columns to the panda dataframe data: ['pmra_sim', 'pmdec_sim']
    """
    ### Simulated pm_ra, pm_dec, and parallax
    pmra_sim, pmdec_sim = np.zeros(len(data)), np.zeros(len(data))

    if noise:
        ### Bin in g magnitude 
        data_g = data['phot_g_mean_mag'].to_numpy()
        min_g, max_g = data_g.min(), data_g.max()
        bins_g = np.arange(min_g, max_g, gmag_bin_size) 
        q_bin_g = np.digitize(data_g, bins_g)-1         

        ### Bin in radial distance from the center        
        center_sky_coord = SkyCoord(ra = sky_p.disc_center[0] * u.deg, dec = sky_p.disc_center[1] * u.deg)
        data_sky_coord = SkyCoord(ra = data['ra'].to_numpy() * u.deg, dec = data['dec'].to_numpy() * u.deg)
        data_r = data_sky_coord.separation(center_sky_coord).value
        bins_r = np.arange(0, np.max(data_r)+rad_bin_size, rad_bin_size)
        q_bin_r = np.digitize(data_r, bins_r)-1     
        
        ### Group the stars according to their g mag and radial position
        df_groupby = pd.DataFrame({'q_bin_g':q_bin_g, 'q_bin_r':q_bin_r,
                                   'pmra_sub':data['pmra_sub'].to_numpy(), 'pmdec_sub':data['pmdec_sub'].to_numpy()}).groupby(by=['q_bin_g', 'q_bin_r'], as_index=False)        
         
        n_bins = 100 ### number of bins to build the PDFs

        ### Loop over groups
        for (g, r) in list(df_groupby.groups.keys()):
            data_group = df_groupby.get_group((g, r))                
            group_index = np.array(list(data_group.index)) # indices for the stars in this group
            n_stars = len(data_group)

            if n_stars > 1:                  
                pdf_pmra, pmra_edges = np.histogram(data_group['pmra_sub'].to_numpy(), bins=n_bins, density=True)
                pdf_pmdec, pmdec_edges = np.histogram(data_group['pmdec_sub'].to_numpy(), bins=n_bins, density=True)

                bin_step_pmra = pmra_edges[1:] - pmra_edges[:-1]
                bin_step_pmdec = pmdec_edges[1:] - pmdec_edges[:-1]

                ### Get the CDFs and interpolate the inverse CDFs
                cdf_pmra = np.cumsum(pdf_pmra)*bin_step_pmra
                inv_cdf_pmra = scipy.interpolate.interp1d(cdf_pmra, pmra_edges[1:])
                pmra_sim[group_index] = 0.9*inv_cdf_pmra(np.random.uniform(cdf_pmra[0], cdf_pmra[-1], n_stars))

                cdf_pmdec = np.cumsum(pdf_pmdec)*bin_step_pmdec
                inv_cdf_pmdec = scipy.interpolate.interp1d(cdf_pmdec, pmdec_edges[1:])
                pmdec_sim[group_index] = 0.9*inv_cdf_pmdec(np.random.uniform(cdf_pmdec[0], cdf_pmdec[-1], n_stars))

            elif n_stars > 0:
                pmra_sim[group_index] = 0.9*np.random.normal(0, sky_p.sigma_pm, n_stars)
                pmdec_sim[group_index] = 0.9*np.random.normal(0, sky_p.sigma_pm, n_stars)
    else:
        print('Skipping noise injection. Setting stars proper motion and parallax to zero.')

    ### Add columns for the simulated pm and parallax
    data.insert(len(data.columns), 'pmra_sim', pmra_sim)
    data.insert(len(data.columns), 'pmdec_sim', pmdec_sim)

    return None


### The velocity of the observer in given by the velocity of the Sun in a reference frame where the galaxy is at rest:
### v_sun = 238 Km/s in the direction l = 270 deg, b = 0 (in Galactic coordinates)
### or alpha = 138 deg, dec = -48.33 deg (in Equatorial coordinates)
### Check i.e. with:
### v_sun_dir = SkyCoord(v_sun_ra*u.rad, v_sun_dec*u.rad)
### v_sun_dir.galactic

_v_sun = 238*math.pow(10,3)*Meter/Second ### magnitude of the Sun velocity
_v_sun_ra, _v_sun_dec = 138.00438151*degree, -48.32963721*degree ### direction of the Sun velocity in equatorial coordnates
_v_obs = _v_sun*fn_3d_unit_vec(np.pi/2-_v_sun_dec, _v_sun_ra)

def fn_obs_vel(ra, dec):
    """
    Returns the observer velocity perpendicular to the line of sight towards the direction (ra, dec).
    """
    theta, phi = np.pi/2 - dec, ra
    u_th = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)]).T
    if np.isscalar(theta):
        u_phi = np.array([-np.sin(phi), np.cos(phi), 0]).T
    else:
        u_phi = np.array([-np.sin(phi), np.cos(phi), np.zeros(len(theta))]).T

    return np.array([np.inner(_v_obs, u_phi), -np.inner(_v_obs, u_th)]).T

### The velocity of the star in a reference frame where the galaxy is at rest is given by 
### the proper motion of the stellar target in the Barycentric Celestial Reference Systems (aligned with ICRS)
### multiplied by the stellar target distance and added to the observer velocity with rispect to the Galactic center,
### as computed by the function fn_obs_vel

def fn_star_vel(v_obs_lens, mu_star_bcrs, distance):
    """
    Returns the star velocity with respect to the galactic center.
    """
    return mu_star_bcrs*distance + v_obs_lens


def fn_signal_inj(data, M_l, r_l, n_lens, lens_pop, sim_sky_patch, n_betat=5, min_beta_t=0.006*degree):
    """
    Injects the lensing signal given by the lenses in lens_pop adding it to the columns pmra_sim and pmdec_sim.
    """
    
    data_ra, data_dec = data['ra'].to_numpy(), data['dec'].to_numpy()
    data_pmra_sim, data_pmdec_sim = np.zeros((len(data))), np.zeros((len(data)))
    
    if n_lens==1:
        vec_lens_array = hp.ang2vec(lens_pop[0], lens_pop[1], lonlat=True) # vector for the location of the lense

        v_obs_lens = fn_obs_vel(lens_pop[0]*degree, lens_pop[1]*degree)
        v_star_lens = fn_star_vel(v_obs_lens, sim_sky_patch.mu_bcrs*mas/Year, sim_sky_patch.distance)
        Dl_over_Di = lens_pop[4]*kpc/sim_sky_patch.distance
        vil_ra = lens_pop[2] - (1-Dl_over_Di)*v_obs_lens[0] - Dl_over_Di*v_star_lens[0]
        vil_dec = lens_pop[3] - (1-Dl_over_Di)*v_obs_lens[1] - Dl_over_Di*v_star_lens[1]
        beta_l = r_l/(lens_pop[4]*kpc)

        ### To find the stars around the lens use a pixelation scale of size approx. beta_l/10 and keep stars withing n_betat*beta_l
        max_beta_l = max(beta_l, min_beta_t)
        n = round(math.log(np.sqrt(np.pi/3)/(0.1*max_beta_l), 2)); nside = 2**n; 
        q_pix = np.asarray(hp.ang2pix(nside, data_ra, data_dec, nest=True, lonlat=True)) 

        ### Find stars around the lens
        nb_lens_i = hp.query_disc(nside, vec_lens_array, n_betat*max_beta_l, inclusive=True, nest=True)
        stars_in = ((q_pix >= nb_lens_i[0]) & (q_pix <= nb_lens_i[-1])) # first reduce the total number of stars
        nb_stars = np.isin(q_pix[stars_in], nb_lens_i, assume_unique=False, invert=False) # keep only stars within the neighboring pixels  

        ### Proper motion template
        beta_it = fn_angular_sep(lens_pop[0]*degree, lens_pop[1]*degree, data_ra[stars_in][nb_stars]*degree, data_dec[stars_in][nb_stars]*degree)
        mura_tilde, mudec_tilde = fn_dipole_mf(beta_l, beta_it)[:2]
        mu_signal = (1-Dl_over_Di)*4*GN*M_l*vil_ra/r_l**2*mura_tilde/(mas/Year) + (1-Dl_over_Di)*4*GN*M_l*vil_dec/r_l**2*mudec_tilde/(mas/Year) # in mas/y

        data_pmra_sim[np.where(stars_in)[0][nb_stars]] += mu_signal[:, 0] # this way of accessing the elements of data_pmra_sim does not make a copy of the array
        data_pmdec_sim[np.where(stars_in)[0][nb_stars]] += mu_signal[:, 1]
        
    else:    
        vec_lens_array = hp.ang2vec(lens_pop[:, 0], lens_pop[:, 1], lonlat=True) # vector for the location of the lenses

        v_obs_lens = fn_obs_vel(lens_pop[:, 0]*degree, lens_pop[:, 1]*degree)
        v_star_lens = fn_star_vel(v_obs_lens, sim_sky_patch.mu_bcrs*mas/Year, sim_sky_patch.distance)
        Dl_over_Di = lens_pop[:, 4]*kpc/sim_sky_patch.distance
        vil_ra = lens_pop[:, 2] - (1-Dl_over_Di)*v_obs_lens[:, 0] - Dl_over_Di*v_star_lens[:, 0]
        vil_dec = lens_pop[:, 3] - (1-Dl_over_Di)*v_obs_lens[:, 1] - Dl_over_Di*v_star_lens[:, 1]
        beta_l = r_l/(lens_pop[:, 4]*kpc)

        ### To find the stars around each lens use a pixelation scale of size approx. max(beta_l)/10 and keep stars withing n_betat*max(beta_l)
        max_beta_l = max(np.max(beta_l), min_beta_t)
        n = round(math.log(np.sqrt(np.pi/3)/(0.1*max_beta_l), 2)); nside = 2**n; 
        q_pix = np.asarray(hp.ang2pix(nside, data_ra, data_dec, nest=True, lonlat=True)) 

        for i, l in enumerate(lens_pop):
            ### Find stars around the lens
            nb_lens_i = hp.query_disc(nside, vec_lens_array[i], n_betat*max_beta_l, inclusive=True, nest=True)
            stars_in = ((q_pix >= nb_lens_i[0]) & (q_pix <= nb_lens_i[-1])) # first reduce the total number of stars
            nb_stars = np.isin(q_pix[stars_in], nb_lens_i, assume_unique=False, invert=False) # keep only stars within the neighboring pixels  

            ### Proper motion template
            beta_it = fn_angular_sep(lens_pop[i, 0]*degree, lens_pop[i, 1]*degree, data_ra[stars_in][nb_stars]*degree, data_dec[stars_in][nb_stars]*degree)
            mura_tilde, mudec_tilde = fn_dipole_mf(beta_l[i], beta_it)[:2]
            mu_signal = (1-Dl_over_Di[i])*4*GN*M_l*vil_ra[i]/r_l**2*mura_tilde/(mas/Year) + (1-Dl_over_Di[i])*4*GN*M_l*vil_dec[i]/r_l**2*mudec_tilde/(mas/Year) # in mas/y
            
            data_pmra_sim[np.where(stars_in)[0][nb_stars]] += mu_signal[:, 0] # this way of accessing the elements of data_pmra_sim does not make a copy of the array
            data_pmdec_sim[np.where(stars_in)[0][nb_stars]] += mu_signal[:, 1]

   
    ### Add signal to the data
    data['pmra_sim'] += data_pmra_sim; data['pmdec_sim'] += data_pmdec_sim; 
    
    return None