import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, run_lefteomcc_calc, get_hbar

basis = 'cc-pvdz'
nfrozen = 0

# Define molecule geometry and basis set
geom = [['H', (0.0, 0.0, -0.8)], 
        ['O', (0.0, 0.0,  0.8)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, charge=-1, unit="Angstrom", symmetry="C2V")

T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')

H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

R, omega_guess = run_guess(H1, H2, o, v, 10, method="ipcis")
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="ipeom3", state_index=[0, 5])
L, omega = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method="left_ipeom3")
#
# Check the results
#
assert np.allclose(Ecorr, -0.216391161750, atol=1.0e-07)
assert np.allclose(omega[0], -0.008144769177, atol=1.0e-07)
assert np.allclose(omega[1], 0.113051103796, atol=1.0e-07)





