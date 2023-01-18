# set allow numpy built with MKL to consume more threads for tensordot
import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

from miniccpy.driver import run_scf, run_cc_calc

basis = 'dz'
nfrozen = 1

# Define molecule geometry and basis set
geom = [['H', (0, 1.515263, -1.058898)], 
        ['H', (0, -1.515263, -1.058898)], 
        ['O', (0.0, 0.0, -0.0090)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen)

T, E_corr = run_cc_calc(fock, g, o, v, method='ccsdtq', out_of_core=True)







