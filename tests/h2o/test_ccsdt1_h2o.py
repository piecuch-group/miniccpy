import numpy as np
from miniccpy.driver import run_scf, run_cc_calc
from miniccpy.pspace import get_active_triples_pspace

basis = '6-31g'
nfrozen = 0

re = 2

# Define molecule geometry and basis set
if re == 1:
    geom = [['H', (0, 1.515263, -1.058898)], 
            ['H', (0, -1.515263, -1.058898)], 
            ['O', (0.0, 0.0, -0.0090)]]
elif re == 2:
    geom = [["O", (0.0, 0.0, -0.0180)],
            ["H", (0.0, 3.030526, -2.117796)],
            ["H", (0.0, -3.030526, -2.117796)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen)

no, nu = fock[o, v].shape

t3_excitations = get_active_triples_pspace(4, 4, no, nu) 

T, E_corr = run_cc_calc(fock, g, o, v, method='ccsdt_p', t3_excitations=t3_excitations)

assert np.allclose(E_corr, -0.30334782, atol=1.0e-07)

