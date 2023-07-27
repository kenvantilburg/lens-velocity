import numpy as np
from utils.my_units import *

def fn_angular_sep_magn_sq(ra1, dec1, ra2, dec2):
    """
    Computes the magnitude of the angular separation vector.
    Angular coordinates given in radians.
    """
    return np.arccos(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra2-ra1), out=np.zeros(len(ra2)), where=((ra1!=ra2) & (dec1!=dec2)) )**2

def fn_angular_sep(ra1, dec1, ra2, dec2):
    """
    Approximate 2d angular separations vector, works for stars close to each other.
    Angular coordinates given in radians.  
    """
    return np.array([(ra1-ra2)*np.cos((dec1+dec2)/2), (dec1-dec2)]).T

def fn_angular_sep_scalar(ra1, dec1, ra2, dec2):
    """
    Approximate magnitude of separations vector, works for stars close to each other.
    Angular coordinates given in radians.    
    """
    return np.sqrt( ((ra1-ra2)*np.cos((dec1+dec2)/2))**2 + (dec1-dec2)**2 )


_rot_matrix = np.array([[1, 0, 0], [0, 0.9174821334228558, 0.39777699135300065], [0, -0.39777699135300065, 0.9174821334228558]])
_ra_offset = 0.05542*arcsec  
def fn_eq_to_ecl_array(ra, dec):
    """
    Function to convert the equatorial coordinates (ra, dec) to ecliptic longitude and latitude according to the Gaia reference frame.
    Works only if ra and dec are numpy arrays. Takes angles in degree and returns in degree.
    Refs.: https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html 
    and section 1.5.3 of https://www.cosmos.esa.int/documents/532822/552851/vol1_all.pdf
    """
    
    ra_s, dec_s = ra*degree + _ra_offset, dec*degree
    x_vec_eq = np.array([np.cos(dec_s)*np.cos(ra_s), np.cos(dec_s)*np.sin(ra_s), np.sin(dec_s)])
    x_vec_ecl = (_rot_matrix @ x_vec_eq).T
    
    ecl_lon, ecl_lat = (np.arctan2(x_vec_ecl[:, 1], x_vec_ecl[:, 0])), np.arctan2(x_vec_ecl[:, 2], np.sqrt(x_vec_ecl[:, 0]**2 + x_vec_ecl[:, 1]**2))
    ecl_lon = ecl_lon + 2*np.pi*np.heaviside(-ecl_lon, 0) ### shift the interval from [-pi, pi] to [0, 2*pi]

    return ecl_lon/degree, ecl_lat/degree

    
    
