# lens-velocity
Code repository for the paper **Astrometric Weak Lensing Constraints on Dark Matter Substructure with Gaia DR3** by Cristina Mondino, Anna-Maria Taki, Andreas Tsantilas, Ken Van Tilburg, and Neal Weiner. Accompanying data files are hosted at [https://users.flatironinstitute.org/~kvantilburg/lens-velocity/](https://users.flatironinstitute.org/~kvantilburg/lens-velocity/).

## Abstract

Small-scale dark matter structures lighter than a billion solar masses are an important probe of primordial density fluctuations and dark matter microphysics. Due to their lack of starlight emission, their only guaranteed signatures are gravitational in nature.

We report on results of a search for astrometric weak lensing by compact dark matter subhalos in the Milky Way with \textit{Gaia} DR3 data. Using a matched-filter analysis to look for correlated imprints of time-domain lensing on the proper motions of background stars in the Magellanic Clouds, we exclude order-unity substructure fractions in halos with masses $M_l$ between $10^7 \, M_\odot$ and $10^9 \, M_\odot$ and sizes of one parsec or smaller.

We forecast that a similar approach based on proper accelerations across the entire sky with data from \textit{Gaia} DR4 may be sensitive to substructure fractions of $f_l \gtrsim 10^{-3}$ in the much lower mass range of $10 \, M_\odot \lesssim M_l \lesssim 3 \times 10^3 \, M_\odot$.

We further propose an analogous technique for \emph{stacked} star-star lensing events in the large-impact-parameter regime. Our first implementation is not yet sufficiently sensitive but serves as a useful diagnostic and calibration tool; future data releases should enable average stellar mass measurements using this stacking method.

## Code
The [code](code/) folder contains various Jupyter notebooks that reproduce the plots in the paper.  Figures from the paper are in [figures](figures/) folder, linked to from the paper. Data is in the [data](data/) folder.

## Authors

- Cristina Mondino (cmondino@perimeterinstitute.ca)
- Anna-Maria Taki (ataki@uoregon.edu)
- Andreas Tsantilas (andreas.tsantilas@nyu.edu )
- Ken Van Tilburg (kenvt@nyu.edu | kvantilburg@flatironinstitute.org)
- Neal Weiner (neal.weiner@nyu.edu)

## Citation

If you use this code, please cite our paper:
```
[put Bibtex here]
```
and you may want to refer to the original papers:
```
@ARTICLE{van-tilburg-2018,
       author = {{Van Tilburg}, Ken and {Taki}, Anna-Maria and {Weiner}, Neal},
        title = "{Halometry from astrometry}",
      journal = {Journal of Cosmology and Astroparticle Physics},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics, High Energy Physics - Phenomenology},
         year = 2018,
        month = jul,
       volume = {2018},
       number = {7},
          eid = {041},
        pages = {041},
          doi = {10.1088/1475-7516/2018/07/041},
archivePrefix = {arXiv},
       eprint = {1804.01991},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2018JCAP...07..041V},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{mondino-2020,
       author = {{Mondino}, Cristina and {Taki}, Anna-Maria and {Van Tilburg}, Ken and {Weiner}, Neal},
        title = "{First Results on Dark Matter Substructure from Astrometric Weak Lensing}",
      journal = {Physical Review Letters},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies, High Energy Physics - Phenomenology},
         year = 2020,
        month = sep,
       volume = {125},
       number = {11},
          eid = {111101},
        pages = {111101},
          doi = {10.1103/PhysRevLett.125.111101},
archivePrefix = {arXiv},
       eprint = {2002.01938},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020PhRvL.125k1101M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
