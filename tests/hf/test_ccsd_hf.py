import numpy as np
from miniccpy.driver import run_scf, run_cc_calc

basis = '6-31g'
nfrozen = 0

geom = [['H', (0.0, 0.0, -0.8)], 
        ['F', (0.0, 0.0,  0.8)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200, unit="Angstrom")

T, E_corr = run_cc_calc(fock, g, o, v, method='ccsd', maxit=80)






