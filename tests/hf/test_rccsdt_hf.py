import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, get_hbar, run_leftcc_calc

basis = '6-31g'
nfrozen = 0

geom = [['H', (0.0, 0.0, -1.0)], 
        ['F', (0.0, 0.0,  1.0)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200, unit="Angstrom", rhf=True)

T, E_corr = run_cc_calc(fock, g, o, v, method='rccsdt', maxit=80)

assert np.allclose(-0.221558736943, E_corr, atol=1.0e-08)




