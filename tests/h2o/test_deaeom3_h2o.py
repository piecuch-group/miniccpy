import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

basis = 'cc-pvdz'
nfrozen = 0

geom = [["O", (0.0, 0.0, -0.0180)],
        ["H", (0.0, 3.030526, -2.117796)],
        ["H", (0.0, -3.030526, -2.117796)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, cartesian=False, charge=2)

T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')
H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')
R, omega_guess = run_guess(H1, H2, o, v, 5, method="deacis", mult=-1, nactu=8)
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="deaeom3", state_index=[0, 1])

#
# Check the results
#
assert np.allclose(Ecorr, -0.161488193574, atol=1.0e-07)
assert np.allclose(omega[0], -1.058733664895, atol=1.0e-07)



