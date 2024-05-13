import numpy as np
from pathlib import Path
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, run_lefteomcc_calc, get_hbar

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eaeom2_chplus():

    fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/chplus.FCIDUMP", 6, 26, 0)

    T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')
    H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

    R, omega_guess = run_guess(H1, H2, o, v, 20, method="eacis")
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method='eaeom2', state_index=[0, 4])
    L, omega = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eaeom2')

    #
    # Check the results
    #
    assert np.allclose(Ecorr, -0.114901980505, atol=1.0e-07)

    assert np.allclose(omega[0], -0.377942659908, atol=1.0e-07)
    assert np.allclose(omega[1], -0.146301569046, atol=1.0e-07)


if __name__ == "__main__":
    test_eaeom2_chplus()





