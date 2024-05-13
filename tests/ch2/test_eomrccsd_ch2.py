import numpy as np
from pathlib import Path
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eomrccsd_ch2():

    fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/ch2-avdz-koch.FCIDUMP", 8, 27, nfrozen=1, rhf=True)

    T, Ecorr = run_cc_calc(fock, g, o, v, method="rccsd")

    H1, H2 = get_hbar(T, fock, g, o, v, method="rccsd")

    R, omega_guess = run_guess(H1, H2, o, v, 30, method="rcis")
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomrccsd", state_index=[0, 1, 2])

    #
    # Check the results
    #
    assert np.allclose(Ecorr, -0.138616456291, atol=1.0e-07)

    assert np.allclose(omega[0], 0.065573488364, atol=1.0e-07)
    assert np.allclose(omega[1], 0.215380798509, atol=1.0e-07)
    assert np.allclose(omega[2], 0.239208033368, atol=1.0e-07)


if __name__ == "__main__":
    test_eomrccsd_ch2()




