# set allow numpy built with MKL to consume more threads for tensordot
import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar


fock, g, e_hf, o, v = run_scf_gamess("chplus.FCIDUMP", 6, 26, 0)

T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')

H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

R0, omega0 = run_guess(H1, H2, o, v, 100)

R, omega, r0 = run_eomcc_calc(R0, omega0, T, H1, H2, o, v, method='eomccsd', state_index=[18, 33, 36, 43], maxit=200)






