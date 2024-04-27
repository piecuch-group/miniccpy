import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, get_hbar, run_leftcc_calc, run_correction

basis = 'cc-pvdz'
nfrozen = 2

geom = [['F', (0.0, 0.0,  2.66816)],
        ['F', (0.0, 0.0, -2.66816)]] 

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, cartesian=True, unit="Bohr", symmetry="D2H")

T, E_corr = run_cc_calc(fock, g, o, v, method='ccsd')
H1, H2 = get_hbar(T, fock, g, o, v, method="ccsd")
L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_ccsd")
delta_T = run_correction(T, L, fock, H1, H2, o, v, method="crcc23")

#
# Check the results
#
assert np.allclose(E_corr, -0.592466290032, atol=1.0e-07)
assert np.allclose(delta_T["A"], -0.03928173204897753, atol=1.0e-07)
assert np.allclose(delta_T["D"], -0.04377671298739536, atol=1.0e-07)







