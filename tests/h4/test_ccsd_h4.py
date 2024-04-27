import numpy as np
from miniccpy.driver import run_scf, run_cc_calc

basis = 'cc-pvdz'
nfrozen = 0
Re = 1

geom = [['H', (-Re, -Re, 0.000)], 
        ['H', (-Re,  Re, 0.000)], 
        ['H', (Re, -Re, 0.000)],
        ['H', (Re,  Re, 0.000)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200)

T, E_corr = run_cc_calc(fock, g, o, v, method='ccsd', maxit=80)

#
# Check the results
#
assert np.allclose(E_corr, -0.084308599849, atol=1.0e-07)






