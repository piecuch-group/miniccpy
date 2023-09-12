import numpy as np
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

fock, g, e_hf, o, v = run_scf_gamess("ch2-avdz-koch.FCIDUMP", 8, 27, nfrozen=0)

T, E_corr = run_cc_calc(fock, g, o, v, method="cc3")

H1, H2 = get_hbar(T, fock, g, o, v, method="ccsdt")

R, omega_guess = run_guess(H1, H2, o, v, 20, method="cis")
#R, omega, r0 = run_eomcc_calc(R, omega_guess, (T[0], T[1]), H1, H2, o, v, method="eomccsd", state_index=[15], max_size=30)
R, omega, r0 = run_eomcc_calc(R, omega, T, H1, H2, o, v, method="eomcc3", state_index=[0], fock=fock, g=g)






