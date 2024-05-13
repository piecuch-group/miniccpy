from pathlib import Path
import numpy as np
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, run_lefteomcc_calc, get_hbar

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eaeom3_chplus():

    fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/chplus.FCIDUMP", 6, 26, 0)

    T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')
    H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

    R, omega_guess = run_guess(H1, H2, o, v, 20, method="eacis")
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method='eaeom3', state_index=[0, 4])
    L, omega = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eaeom3')

    #
    # Check the results
    #
    assert np.allclose(Ecorr, -0.114901980505, atol=1.0e-07)

    assert np.allclose(omega[0], -0.378155697986, atol=1.0e-07)
    assert np.allclose(omega[1], -0.145524587235, atol=1.0e-07)

if __name__ == "__main__":
    test_eaeom3_chplus()





