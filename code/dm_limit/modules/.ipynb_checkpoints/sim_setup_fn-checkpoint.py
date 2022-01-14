import numpy as np
from scipy import integrate
from scipy.optimize import fsolve
import scipy.stats as st
from scipy.spatial.transform import Rotation as R
import sys
from my_units import *

_rs_MW = 18*kpc ### r_s parameter of the NFW profile of the Milky Way (scale radius)
_rho_s_MW = 0.003*MSolar/pc**3 ### rho_s  parameter of the NFW profile of the Milky Way (scale density)
_d_Sun = 8.29*kpc ### distance of the Sun from the Galactic Center

def fn_rho_dm(dist, l, b): ### works with scalar quantities
    """
    Returns the Dark Matter energy density at distance dist and in the direction (l, b), given in Galactic coordinates.
    """
    r_vec_sun = np.array([0, _d_Sun, 0]) ### 3d vector position of the Sun wrt the Galactic Center    
    r_vec = dist*np.array([np.sin(l)*np.cos(b), np.cos(l)*np.cos(b), np.sin(b)]) - r_vec_sun ### 3d vector wrt the Galactic Center
    r_over_rs = np.linalg.norm(r_vec/_rs_MW)
    
    return 4*_rho_s_MW/(r_over_rs*(1+r_over_rs)**2)

def fn_rho_dm_array(dist, l, b): ### works with numpy arrays 
    """
    Returns the Dark Matter energy density at distance dist and in the direction (l, b), given in Galactic coordinates.
    """
    r_vec_sun = np.full((len(dist), 3), np.array([0, _d_Sun, 0])) ### 3d vector position of the Sun wrt the Galactic Center  
    r_vec = np.array([dist*np.sin(l)*np.cos(b) - r_vec_sun[:, 0],
                      dist*np.cos(l)*np.cos(b) - r_vec_sun[:, 1],
                      dist*np.sin(b) - r_vec_sun[:, 2]]).T  ### 3d vector wrt the Galactic Center
    r_over_rs = np.linalg.norm(r_vec/_rs_MW, axis=1)
    
    return 4*_rho_s_MW/(r_over_rs*(1+r_over_rs)**2)


def fn_n_lens(M_l, f_l, dist, l, b, delta_omega):
    """
    Returns the average number of lenses with mass M_l and fractional abundance f_l in front of a stellar target centered at Galactic Coordinates (l, b) and covering a solid angle delta_omega, up to a distance dist.
    """
    integrand = lambda dist: dist**2*fn_rho_dm(dist, l, b) ### function to be integrated over distance        
    return delta_omega*f_l*integrate.quad(integrand, 0, dist)[0]/M_l

def fn_n_lens_tot(M_l, f_l, dist, sky_patches, dist_max=False):
    """
    Returns the total number of lenses in front of all the sky patches used in the analysis up to a distance dist.
    If dist_max=True, the distance is set to the distance of the stellar target for each patch.
    """
    n_lens_tot = 0
        
    for i in range(len(sky_patches)):
        if dist_max==False:
            n_lens_tot += fn_n_lens(M_l, f_l, dist, sky_patches[i].center_l*degree, sky_patches[i].center_b*degree, sky_patches[i].delta_omega)    
        else:
            n_lens_tot += fn_n_lens(M_l, f_l, sky_patches[i].distance, sky_patches[i].center_l*degree, sky_patches[i].center_b*degree, sky_patches[i].delta_omega)    
            
    return n_lens_tot

def fn_beta_t_opt(M_l, r_l, f_l, sky_patches):
    """
    Find the beta_t optimal for the given lens population parameters and the given stellar targets used in the analysis.
    """
    n_lens_opt = 3 ### number of optimal lenses with beta_t >= beta_t_optimal
    n_lens_max = fn_n_lens_tot(M_l, f_l, 0, sky_patches, dist_max=True) ### total number of lenses in front of the stellar targets
        
    if n_lens_max>=n_lens_opt:
        d_min_list = np.zeros((len(sky_patches)))        
        for i in range(len(sky_patches)):
            d_min_list[i] = (3*M_l/(sky_patches[i].delta_omega*_rho_s_MW))**(1/3)
        dist_solve = lambda d : (fn_n_lens_tot(M_l, f_l, d, sky_patches, dist_max=False) - n_lens_opt)
        d_opt = fsolve(dist_solve, np.min(d_min_list))[0] # solve for dist_solve==0, starting from the minimum distance at which there is one lens
    else:
        ### take the average of the distance to each patch on the sky
        d_opt = 0
        for i in range(len(sky_patches)):
            d_opt += sky_patches[i].distance
        d_opt = d_opt/len(sky_patches)
        
    return r_l/d_opt ### optimal beta_t in radians


def fn_beta_t_opt_list(beta_t_optimal, beta_t_list):
    """
    Find the beta_t values from beta_t_list closer to the optimal beta_t
    """

    ### Find the beta_t values from the beta_t_list closest to the optimal beta_t
    beta_t_opt_ind = np.argmin(np.abs(np.array(beta_t_list) - beta_t_optimal))
    if beta_t_opt_ind==0:
        beta_t_opt_list = beta_t_list[:beta_t_opt_ind+2]
    elif beta_t_opt_ind==(len(beta_t_list)-1):
        beta_t_opt_list = beta_t_list[beta_t_opt_ind-1:]
    else:
        beta_t_opt_list = beta_t_list[beta_t_opt_ind-1:beta_t_opt_ind+2]
    
    return beta_t_opt_list 


_v0_ss = 238*1000*Meter/Second ### Solar System velocity, i.e. the observer velocity wrt the DM halo
_sigma_vl = 166*1000*Meter/Second #_v0_ss/math.sqrt(2) ### Dark Matter velocity dispersion

def fn_3d_unit_vec(th, phi):
    """Unit vector in 3d"""
    return np.array([np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), np.cos(th)]).T

def fn_lens_population(M_l, f_l, sky_patch, n_lens_max):
    """Generate a random population of lenses in front of the stellar target"""
    ### Random number of lenses in front of the stellar target with Poisson distribution
    n_lens_avg = fn_n_lens(M_l, f_l, sky_patch.distance, sky_patch.center_l*degree, sky_patch.center_b*degree, sky_patch.delta_omega)    
    n_lens = np.random.poisson(n_lens_avg)
    
    ### Probability distribution function for the polar angle theta within the region covered by the stellar target
    class pdf_theta(st.rv_continuous):
        def _pdf(self, th):
            return np.sin(th)/(1-np.cos(sky_patch.disc_radius))
    theta_dist = pdf_theta(a=0, b=sky_patch.disc_radius, name='pdf_theta')

    ### Random lens position in front of the stellar target
    theta_lens = theta_dist.rvs(size=n_lens)
    phi_lens = np.random.uniform(0, 2*np.pi, n_lens)
    ### Rotation in the direction of the stellar target
    vec_center_patch = fn_3d_unit_vec(np.pi/2-sky_patch.center_dec*degree, sky_patch.center_ra*degree)
    rot_unit_vector = np.cross(np.array([0, 0, 1]), vec_center_patch) ### direction of the rotation axis, must be normalized
    rot_unit_vector = rot_unit_vector/np.sqrt(rot_unit_vector[0]**2 + rot_unit_vector[1]**2 + rot_unit_vector[2]**2)
    rot_matrix = R.from_rotvec(np.arccos(np.dot(np.array([0, 0, 1]), vec_center_patch)) * rot_unit_vector)

    rot_lens_vectors = rot_matrix.apply(fn_3d_unit_vec(theta_lens, phi_lens))
    lens_ra = (np.arctan2(rot_lens_vectors[:, 1], rot_lens_vectors[:, 0]))/degree
    lens_dec = (np.pi/2-np.arctan2(np.sqrt(rot_lens_vectors[:, 0]**2 + rot_lens_vectors[:, 1]**2), rot_lens_vectors[:, 2]))/degree 
        
    ### Random lens velocities
    vl_ra = np.random.normal(0, _sigma_vl, n_lens)
    vl_dec = np.random.normal(0, _sigma_vl, n_lens)
        
    ### Probability distribution function for the lens distance, assuming the DM energy density along the line of sight towards the center of the stellar target    
    class pdf_dist(st.rv_continuous):
        def _pdf(self, d):
            return 1/n_lens_avg*sky_patch.delta_omega*f_l*d**2*fn_rho_dm(d, sky_patch.center_l*degree, sky_patch.center_b*degree)/M_l 
    d_dist = pdf_dist(a=0, b=sky_patch.distance, name='pdf_dist')
    
    dl = d_dist.rvs(size=n_lens)/kpc
    
    if n_lens==1:
        lens_pop = np.array([lens_ra, lens_dec, vl_ra, vl_dec, dl]).T[0]
    else:
        lens_pop = np.array([lens_ra, lens_dec, vl_ra, vl_dec, dl]).T  

        ### Sort lenses based on their distance and keep only the n_lens_max closest ones
        if n_lens > n_lens_max:
            print(n_lens, 'lenses. Retaining only the closest', n_lens_max)
            sys.stdout.flush()
            lens_ind_sort = np.argsort(lens_pop[:, -1])
            lens_pop = lens_pop[lens_ind_sort[:n_lens_max]]
        
    return n_lens, lens_pop