import numpy as np
from miniccpy.driver import run_scf, run_cc_calc

basis = '6-31g'
nfrozen = 0
# Define molecule geometry and basis set
geom = [['F', (0.0, 0.0, -2.66816)], 
        ['F', (0.0, 0.0,  2.66816)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, unit="Bohr")

T, E_corr = run_cc_calc(fock, g, o, v, method="cc3")

