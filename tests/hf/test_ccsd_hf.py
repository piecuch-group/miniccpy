import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, get_hbar, run_leftcc_calc

basis = 'cc-pvdz'
nfrozen = 0

geom = [['H', (0.0, 0.0, -1.0)], 
        ['F', (0.0, 0.0,  1.0)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200, unit="Angstrom")

T, E_corr = run_cc_calc(fock, g, o, v, method='ccsd', maxit=80)
H1, H2 = get_hbar(T, fock, g, o, v, method="ccsd")
L = run_leftcc_calc(T, H1, H2, o, v, method="left_ccsd")






