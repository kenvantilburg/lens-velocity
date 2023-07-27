import numpy as np
from my_units import *
import astropy.units as u
from astropy.coordinates import SkyCoord

class sky_patch:
    """Class defining the stellar target properties. Notice that this works only for a disk on the sky."""
    def __init__(self, center_ra, center_dec, disc_radius, distance, data_file_name, mu_bcrs=np.array([0,0]), pm_esc = 0, sigma_pm = 0):
        self.center_ra = center_ra
        self.center_dec = center_dec
        self.disc_center = np.array([center_ra, center_dec])
        self.disc_radius = disc_radius
        self.distance = distance # distance from the observer
        ### Galactic coordinates of the center (in deg)
        self.center_l = SkyCoord(center_ra*u.deg, center_dec*u.deg, frame = 'icrs').galactic.l.radian/degree 
        self.center_b = SkyCoord(center_ra*u.deg, center_dec*u.deg, frame = 'icrs').galactic.b.radian/degree 
        ### Solid angle covered by the stellar target (in radians)
        self.delta_omega = 2*np.pi*(1-math.cos(disc_radius)) 
        ### Proper motion of the stellar target in the Barycentric Celestial Reference Systems (aligned with ICRS)
        self.mu_bcrs = mu_bcrs # in mas/y
        self.data_file_name = data_file_name
        ### Magnitude of the escape proper motion for stars in the stellar target
        self.pm_esc = pm_esc
        ### Intrinsic pm dispersion in the stars - for noise injection of the very bright stars
        self.sigma_pm = sigma_pm     