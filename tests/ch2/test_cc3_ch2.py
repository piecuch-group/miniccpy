import numpy as np
from pathlib import Path
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_lefteomcc_calc

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/ch2-avdz-koch.FCIDUMP", 8, 27, nfrozen=0)

T, Ecorr = run_cc_calc(fock, g, o, v, method="cc3")

H1, H2 = get_hbar(T, fock, g, o, v, method="cc3")

R, omega_guess = run_guess(H1, H2, o, v, 20, method="cisd", mult=1, nacto=2, nactu=4)
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomcc3", state_index=[0, 1, 2, 3, 4], fock=fock, g=g)

#
# Check the results
#
assert np.allclose(Ecorr, -0.143442285157, atol=1.0e-07)
assert np.allclose(omega[0], 0.065711012562, atol=1.0e-07) 
assert np.allclose(omega[1], 0.188372852891, atol=1.0e-07) 
assert np.allclose(omega[2], 0.215282206248, atol=1.0e-07) 
assert np.allclose(omega[3], 0.239241550422, atol=1.0e-07) 
assert np.allclose(omega[4], 0.283715639343, atol=1.0e-07) 





