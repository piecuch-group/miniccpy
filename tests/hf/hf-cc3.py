# set allow numpy built with MKL to consume more threads for tensordot
import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

import numpy as np
from miniccpy.driver import run_scf, run_cc_calc

basis = '6-31g'
nfrozen = 0
# Define molecule geometry and basis set
geom = [['H', (0.0, 0.0, -0.8)], 
        ['F', (0.0, 0.0,  0.8)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, unit="Angstrom", symmetry="C2V")

T, E_corr = run_cc_calc(fock, g, o, v, method="cc3")

assert np.isclose(E_corr, -0.178932834216145, atol=1e-9)





