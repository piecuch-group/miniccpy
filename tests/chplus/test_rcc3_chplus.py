import numpy as np
from pathlib import Path
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_rcc3_chplus():

    fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/chplus.FCIDUMP", 6, 26, 0, rhf=True)

    T, E_corr = run_cc_calc(fock, g, o, v, method='rcc3')

    #
    # Check the results
    #
    assert np.allclose(-0.116362571924, E_corr, atol=1.0e-08)

if __name__ == "__main__":
    test_rcc3_chplus()