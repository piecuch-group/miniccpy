import numpy as np
from pathlib import Path
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/ch2-avdz-koch.FCIDUMP", 8, 27, nfrozen=0)

T, E_corr = run_cc_calc(fock, g, o, v, method="ccsd")

H1, H2 = get_hbar(T, fock, g, o, v, method="ccsd")

R, omega_guess = run_guess(H1, H2, o, v, 30, method="cis", mult=1)
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomccsd", state_index=[x for x in range(10)], max_size=20)






