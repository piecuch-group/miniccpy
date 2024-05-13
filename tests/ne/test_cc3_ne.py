import numpy as np
from pathlib import Path
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_leftcc_calc, run_lefteomcc_calc

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_cc3_ne():

    fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/ne-avdz.FCIDUMP", 10, 18, nfrozen=0)

    T, E_corr = run_cc_calc(fock, g, o, v, method="cc3")

    H1, H2 = get_hbar(T, fock, g, o, v, method="cc3")

    L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_cc3", g=g)

    R, omega_guess = run_guess(H1, H2, o, v, 10, method="cis", mult=1)
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomcc3", state_index=[0], fock=fock, g=g)
    L, omega = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method="left_eomcc3", fock=fock, g=g)

    #
    # Check the results
    #
    assert np.allclose(E_corr, -0.193443908386, atol=1.0e-07)
    assert np.allclose(omega[0], 0.603109465107, atol=1.0e-07)

if __name__ == "__main__":
    test_cc3_ne()