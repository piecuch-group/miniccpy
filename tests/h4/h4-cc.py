# set allow numpy built with MKL to consume more threads for tensordot
import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

from miniccpy.driver import run_scf, run_cc_calc

basis = 'cc-pvdz'
nfrozen = 0
Re = 5

geom = [['H', (-Re, -Re, 0.000)], 
        ['H', (-Re,  Re, 0.000)], 
        ['H', ( Re, -Re, 0.000)], 
        ['H', ( Re,  Re, 0.000)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200, convergence=1.0e-013)

T, E_corr = run_cc_calc(fock, g, o, v, method='ccsd', maxit=80, convergence=1.0e-08, shift=0.8, diis_size=8, use_quasi=True)







