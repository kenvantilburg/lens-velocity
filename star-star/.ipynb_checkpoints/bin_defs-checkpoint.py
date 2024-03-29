import numpy as np

### Spatial pixelation with healpy pixels at level 8 (as used in EDR3)
n = 4
fac_source_id = 2**(59-2*n) # factorization used to extract the healpy bin from the source id

# distance bins
bins_dist_neg = np.concatenate([[-100000, 0], np.logspace(np.log10(1000), np.log10(10000),5), [200000]]) # including a nagative distance bin for the stars that overflow
bins_dist = np.concatenate([[0], np.logspace(np.log10(1000), np.log10(10000),5), [200000]])

# g-magnitude bins
bins_G = np.arange(5,23,1)# np.arange(10,24,1)

# radial separation bins
step = 0.3 # Size of bin
start = 0.0 # Starting bin in arcsec
end = 3 + step  #3 + 2 * step # Ending bin in arcsec
bins_bil = np.arange(start, end, step)