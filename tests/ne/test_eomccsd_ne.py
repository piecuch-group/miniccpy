import numpy as np
from pathlib import Path
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eomccsd_ne():

    fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/ne-avdz.FCIDUMP", 10, 18, nfrozen=0)

    T, E_corr = run_cc_calc(fock, g, o, v, method="ccsd")

    H1, H2 = get_hbar(T, fock, g, o, v, method="ccsd")

    R, omega_guess = run_guess(H1, H2, o, v, 30, method="cis")
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomccsd", state_index=[9], max_size=20)

    #
    # Check the results
    #
    assert np.allclose(E_corr, -0.192193526548, atol=1.0e-07)
    assert np.allclose(omega[0], 0.593793701486, atol=1.0e-07)

if __name__ == "__main__":
    test_eomccsd_ne()