import numpy as np
from pathlib import Path
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_rcc3_chplus():

    fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/chplus.FCIDUMP", 6, 26, 0, rhf=True)

    T, E_corr = run_cc_calc(fock, g, o, v, method='rcc3')
    H1, H2 = get_hbar(T, fock, g, o, v, method="rcc3")

    R0, omega0 = run_guess(H1, H2, o, v, 10, method="rcis")
    R, omega, r0 = run_eomcc_calc(R0, omega0, T, H1, H2, o, v, method='eomrcc3', state_index=[0, 2], fock=fock, g=g)

    #
    # Check the results
    #
    assert np.allclose(-0.116362571924, E_corr, atol=1.0e-08)
    assert np.allclose(0.119139155347, omega[0], atol=1.0e-07)
    assert np.allclose(0.497618740115, omega[1], atol=1.0e-07)

if __name__ == "__main__":
    test_rcc3_chplus()
