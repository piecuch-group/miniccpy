# set allow numpy built with MKL to consume more threads for tensordot
import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

basis = 'dz'
nfrozen = 0

# Define molecule geometry and basis set
geom = [['H', (0, 1.515263, -1.058898)], 
        ['H', (0, -1.515263, -1.058898)], 
        ['O', (0.0, 0.0, -0.0090)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen)

T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')

H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

R, omega_guess = run_guess(H1, H2, o, v, 5, method="eacis")
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eaeom2", state_index=[1, 2, 3])






