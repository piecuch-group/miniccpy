# set allow numpy built with MKL to consume more threads for tensordot
import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

import numpy as np
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

fock, g, e_hf, o, v = run_scf_gamess("ne-avdz.FCIDUMP", 10, 18, nfrozen=0)

T, E_corr = run_cc_calc(fock, g, o, v, method="cc3")

H1, H2 = get_hbar(T, fock, g, o, v, method="ccsdt")

R, omega_guess = run_guess(H1, H2, o, v, 30, method="cis")
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomcc3", state_index=[9], max_size=20, fock=fock, g=g)
