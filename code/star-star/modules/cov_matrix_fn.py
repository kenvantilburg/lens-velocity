import numpy as np
import pandas as pd

def fn_cov_pm_eff(df):
    """
    Given a panda data frame df, returns the effective covariance matrix for (pmra, pmdec).
    """
    return np.array([[df['pmra_eff_error']**2, df['pmra_pmdec_eff_corr']*df['pmra_eff_error']*df['pmdec_eff_error']], 
                     [df['pmra_pmdec_eff_corr']*df['pmra_eff_error']*df['pmdec_eff_error'], df['pmdec_eff_error']**2]]).T
    
    
