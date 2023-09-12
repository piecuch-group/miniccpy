import numpy as np
from pathlib import Path
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/chplus.FCIDUMP", 6, 26, 0)

T, Ecorr  = run_cc_calc(fock, g, o, v, method='cc3')
H1, H2 = get_hbar(T, fock, g, o, v, method='ccsdt')
R, omega_guess = run_guess(H1, H2, o, v, 20, method="cis", mult=1)
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method='eomcc3', state_index=[0, 2, 4], fock=fock, g=g)


