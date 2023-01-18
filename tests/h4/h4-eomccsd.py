# set allow numpy built with MKL to consume more threads for tensordot
import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

from miniccpy.driver import run_scf, run_cc_calc, run_eomcc_calc, get_hbar

basis = 'dz'
nfrozen = 0 

# Define molecule geometry and basis set
geom = [['H', (-1.000, -1.000/2, 0.000)], 
        ['H', (-1.000,  1.000/2, 0.000)], 
        ['H', ( 1.000, -1.000/2, 0.000)], 
        ['H', ( 1.000,  1.000/2, 0.000)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen)

T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')

H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

nroot = 5
R, omega, r0 = run_eomcc_calc(T, fock, g, H1, H2, o, v, nroot, method='eomccsd')






