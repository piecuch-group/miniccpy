import numpy as np
from miniccpy.driver import run_scf, run_cc_calc

basis = 'dz'
nfrozen = 0

# Define molecule geometry and basis set
geom = [['H', ( 2.4143135624,  1.000, 0.000)], 
        ['H', (-2.4143135624,  1.000, 0.000)],
        ['H', ( 2.4143135624, -1.000, 0.000)],
        ['H', (-2.4143135624, -1.000, 0.000)],
        ['H', ( 1.000,  2.4142135624, 0.000)],
        ['H', (-1.000,  2.4142135624, 0.000)],
        ['H', ( 1.000, -2.4142135624, 0.000)],
        ['H', (-1.000, -2.4142135624, 0.000)],
        ]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen)

T, E_corr = run_cc_calc(fock, g, o, v, method='ccsd')

#
# Check the results
#
assert np.allclose(E_corr, -0.155467646059, atol=1.0e-07)






