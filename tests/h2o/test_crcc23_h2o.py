import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, get_hbar, run_leftcc_calc, run_correction

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

T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')
H1, H2 = get_hbar(T, fock, g, o, v, method="ccsd")
L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_ccsd")
delta_T = run_correction(T, L, fock, H1, H2, o, v, method="crcc23")

#
# Check the results
#
assert np.allclose(Ecorr, -0.291219152750, atol=1.0e-07)
assert np.allclose(delta_T["A"], -0.009907050495912655, atol=1.0e-07)
assert np.allclose(delta_T["D"], -0.01333695816624863, atol=1.0e-07)