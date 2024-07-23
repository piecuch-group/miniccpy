"""
Compares the result of CC3 against the result obtained from Psi4
for the lowest-lying singlet state of the H2O molecule at the Re and
2Re structures, obtained from JCP 104, 8007 (1996).
"""

import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_leftcc_calc, run_lefteomcc_calc

def test_cc3_h2o():

    basis = 'cc-pvdz'
    nfrozen = 0

    re = 1

    # Define molecule geometry and basis set
    if re == 1:
        geom = [['H', (0, 1.515263, -1.058898)],
                ['H', (0, -1.515263, -1.058898)],
                ['O', (0.0, 0.0, -0.0090)]]
    elif re == 2:
        geom = [["O", (0.0, 0.0, -0.0180)],
                ["H", (0.0, 3.030526, -2.117796)],
                ["H", (0.0, -3.030526, -2.117796)]]

    fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, rhf=True)

    T, Ecorr  = run_cc_calc(fock, g, o, v, method='rcc3')

    H1, H2 = get_hbar(T, fock, g, o, v, method='rcc3')


    R, omega_guess = run_guess(H1, H2, o, v, 5, method="rcis")
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomrcc3", state_index=[0], fock=fock, g=g)

    #
    # Check the results
    #

if __name__ == "__main__":
    test_cc3_h2o()




