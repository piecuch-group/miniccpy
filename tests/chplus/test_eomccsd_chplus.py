from pathlib import Path
import numpy as np
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_lefteomcc_calc

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eomccsd_chplus():

    fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/chplus.FCIDUMP", 6, 26, 0)

    T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')

    H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

    R0, omega0 = run_guess(H1, H2, o, v, 20, method="cis")

    R, omega, r0 = run_eomcc_calc(R0, omega0, T, H1, H2, o, v, method='eomccsd', state_index=[17], maxit=80)
    L, omega = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eomccsd', maxit=80)

    #
    # Check the results
    #
    assert np.allclose(Ecorr, -0.114901980505, atol=1.0e-07)
    assert np.allclose(omega[0], 0.499068731590, atol=1.0e-07)

if __name__ == "__main__":
    test_eomccsd_chplus()


