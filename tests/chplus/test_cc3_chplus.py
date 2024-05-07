import numpy as np
from pathlib import Path
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_leftcc_calc, run_lefteomcc_calc

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/chplus.FCIDUMP", 6, 26, 0)

T, Ecorr = run_cc_calc(fock, g, o, v, method='cc3')
H1, H2 = get_hbar(T, fock, g, o, v, method='cc3')
L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_cc3", g=g)
R, omega_guess = run_guess(H1, H2, o, v, 20, method="cisd", mult=1, nacto=6, nactu=6)
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method='eomcc3', state_index=[0, 2], fock=fock, g=g)
L, omega = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method="left_eomcc3", fock=fock, g=g)

#
# Check the results
#
assert np.allclose(Ecorr, -0.116362571924, atol=1.0e-07)
assert np.allclose(omega[0], 0.119139155347, atol=1.0e-07)
assert np.allclose(omega[1], 0.196038817160, atol=1.0e-07)
