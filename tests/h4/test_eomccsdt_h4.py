import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_eomcc_calc, get_hbar, run_guess, run_lefteomcc_calc, run_leftcc_calc

basis = 'dz'
nfrozen = 0 

# Define molecule geometry and basis set
geom = [['H', (-1.000, -1.000/2, 0.000)], 
        ['H', (-1.000,  1.000/2, 0.000)], 
        ['H', ( 1.000, -1.000/2, 0.000)], 
        ['H', ( 1.000,  1.000/2, 0.000)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen)

T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsdt')

H1, H2 = get_hbar(T, fock, g, o, v, method='ccsdt')

R, omega_guess = run_guess(H1, H2, o, v, 10, method="cis")
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method='eomccsdt', state_index=[0, 3, 6])
L, omega_left = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eomccsdt')

#
# Check the results
#
assert np.allclose(Ecorr, -0.045886569158, atol=1.0e-07)
assert np.allclose(omega[0], 0.282407233995, atol=1.0e-07)
assert np.allclose(omega[1], 0.346390963495, atol=1.0e-07)
assert np.allclose(omega[2], 0.541791372066, atol=1.0e-07)



