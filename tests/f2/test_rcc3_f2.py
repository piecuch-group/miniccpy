import numpy as np
from miniccpy.driver import run_scf, run_cc_calc

basis = '6-31g'
nfrozen = 0
# Define molecule geometry and basis set
geom = [['F', (0.0, 0.0, -2.66816)], 
        ['F', (0.0, 0.0,  2.66816)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, unit="Bohr", rhf=True)

T, E_corr = run_cc_calc(fock, g, o, v, method="rcc3")

assert np.allclose(-0.500830858037, E_corr, atol=1.0e-08)

