from pathlib import Path
import numpy as np
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_lefteomcc_calc

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/chplus.FCIDUMP", 6, 26, 0)

T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')

H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

R0, omega0 = run_guess(H1, H2, o, v, 100, method="cis")

R, omega, r0 = run_eomcc_calc(R0, omega0, T, H1, H2, o, v, method='eomccsd', state_index=[17, 32, 37, 42], maxit=200)
L, omega_left = run_eomcc_calc(R, omega, T, H1, H2, o, v, method='left_eomccsd', maxit=200)






