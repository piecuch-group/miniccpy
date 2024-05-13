import numpy as np
from pathlib import Path
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eomccsd_ch2():

    fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/ch2-avdz-koch.FCIDUMP", 8, 27, nfrozen=0)

    T, E_corr = run_cc_calc(fock, g, o, v, method="ccsd")

    H1, H2 = get_hbar(T, fock, g, o, v, method="ccsd")

    R, omega_guess = run_guess(H1, H2, o, v, 30, method="cisd", mult=1, nacto=4, nactu=8)
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomccsd", state_index=[0, 1, 2, 3, 4], max_size=20)

    #
    # Check the results
    #
    assert np.allclose(E_corr, -0.140400621098, atol=1.0e-07)
    assert np.allclose(omega[0], 0.065435316381, atol=1.0e-07)
    assert np.allclose(omega[1], 0.224588635117, atol=1.0e-07)
    assert np.allclose(omega[2], 0.215300416665, atol=1.0e-07)
    assert np.allclose(omega[3], 0.239230666971, atol=1.0e-07)
    assert np.allclose(omega[4], 0.283506868176, atol=1.0e-07)

if __name__ == "__main__":
    test_eomccsd_ch2()
