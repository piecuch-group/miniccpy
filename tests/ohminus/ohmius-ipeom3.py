# set allow numpy built with MKL to consume more threads for tensordot
import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

basis = 'cc-pvdz'
nfrozen = 0

# Define molecule geometry and basis set
geom = [['H', (0.0, 0.0, -0.8)], 
        ['O', (0.0, 0.0,  0.8)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, charge=-1, unit="Angstrom", symmetry="C2V")

T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')

H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

R, omega_guess = run_guess(H1, H2, o, v, 10, method="ipcis")
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="ipeom3", state_index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], max_size=40)






