# set allow numpy built with MKL to consume more threads for tensordot
import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

import numpy as np
from miniccpy.driver import run_scf, run_cc_calc

basis = '6-31g'
nfrozen = 0
# Define molecule geometry and basis set
geom = [['F', (0.0, 0.0, 0.0)], 
        ['F', (0.0, 0.0, 1.6)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, unit="Angstrom")

T, E_corr = run_cc_calc(fock, g, o, v, method="cc3")

