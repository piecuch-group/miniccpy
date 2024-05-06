from pathlib import Path
import numpy as np
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_lefteomcc_calc, run_eom_correction

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/chplus.FCIDUMP", 6, 26, 0)

T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')

H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

R0, omega0 = run_guess(H1, H2, o, v, 15, method="cisd", nacto=6, nactu=6, mult=1)

R, omega, r0 = run_eomcc_calc(R0, omega0, T, H1, H2, o, v, method='eomccsd', state_index=[0, 5, 7], maxit=80)
L, omega_left = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eomccsd', maxit=80)

delta_T = []
for i in range(len(R)):
    delta_T.append(run_eom_correction(T, R[i], L[i], r0[i], omega[i], fock, H1, H2, o, v, method="creomcc23"))
    
#
# Check the results
#
assert np.allclose(Ecorr, -0.114901981160, atol=1.0e-07)

assert np.allclose(omega[0], 0.119828871481, atol=1.0e-07)
assert np.allclose(delta_T[0]["A"], -0.0016296077512572497, atol=1.0e-07)
assert np.allclose(delta_T[0]["D"], -0.0022877875056616106, atol=1.0e-07)

assert np.allclose(omega[1], 0.289860992958, atol=1.0e-07)
assert np.allclose(delta_T[1]["A"], -0.0237493047609694, atol=1.0e-07)
assert np.allclose(delta_T[1]["D"], -0.034803211361375874, atol=1.0e-07)

assert np.allclose(omega[2], 0.334743390593, atol=1.0e-07) 
assert np.allclose(delta_T[2]["A"], -0.012984809381926702, atol=1.0e-07)
assert np.allclose(delta_T[2]["D"], -0.018321422678481428, atol=1.0e-07)





