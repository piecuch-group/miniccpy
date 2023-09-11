# set allow numpy built with MKL to consume more threads for tensordot
import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

basis = '6-31g'
nfrozen = 0
# Define molecule geometry and basis set
geom = [['H', (0.0, 0.0, -0.8)], 
        ['F', (0.0, 0.0,  0.8)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, unit="Angstrom", symmetry="C2V")

T, E_corr = run_cc_calc(fock, g, o, v, method="cc3")

H1, H2 = get_hbar((T[0],T[1]), fock, g, o, v, method="ccsd")
R, omega_guess = run_guess(H1, H2, o, v, 30, method="cis")
R, omega, r0 = run_eomcc_calc(R, omega_guess, (T[0],T[1]), H1, H2, o, v, method="eomccsd", state_index=[9, 26], max_size=50)

#H1, H2 = get_hbar(T, fock, g, o, v, method="ccsdt")
#R, omega_guess = run_guess(H1, H2, o, v, 10, method="cis")
#R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomcc3", state_index=[9], fock=fock, g=g, max_size=20)

assert np.isclose(E_corr, -0.178932834216145, atol=1e-9)





