import numpy as np
from pathlib import Path
from miniccpy.driver import (run_scf_gamess, 
                             run_cc_calc, run_leftcc_calc,
                             get_hbar,
                             run_guess, run_eomcc_calc, run_lefteomcc_calc, 
                             run_correction, run_eom_correction)

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_creom23_ch2():

    fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/ch2-avdz-koch.FCIDUMP", 8, 27, nfrozen=0)

    T, Ecorr = run_cc_calc(fock, g, o, v, method="ccsd")

    H1, H2 = get_hbar(T, fock, g, o, v, method="ccsd")

    L = run_leftcc_calc(T, fock, H1, H2, o, v, method='left_ccsd')
    delta_T = run_correction(T, L, fock, H1, H2, o, v, method="crcc23")

    R, omega_guess = run_guess(H1, H2, o, v, 20, method="cisd", mult=1, nacto=2, nactu=4)
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomccsd", state_index=[0, 1, 2, 3, 4])
    L, omega_left = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eomccsd')

    for i in range(len(R)):
        delta_T = run_eom_correction(T, R[i], L[i], r0[i], omega[i], fock, H1, H2, o, v, method="creomcc23")

    #
    # Check the results
    #

if __name__ == "__main__":
    test_creom23_ch2()
