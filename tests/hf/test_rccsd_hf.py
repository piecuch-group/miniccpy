import numpy as np
from miniccpy.driver import run_scf, run_cc_calc

basis = 'cc-pvdz'
nfrozen = 0

geom = [['H', (0.0, 0.0, -1.0)], 
        ['F', (0.0, 0.0,  1.0)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200, unit="Angstrom", rhf=True, cartesian=False)

T, E_corr = run_cc_calc(fock, g, o, v, method='r-ccsd', maxit=80)

assert np.allclose(-0.277969253916, E_corr, atol=1.0e-08)







