"""
Compares the result of CC3 against the result obtained from Psi4
for the lowest-lying singlet state of the H2O molecule at the Re and
2Re structures, obtained from JCP 104, 8007 (1996).
"""

import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_correction

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

T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')
delta_T = run_correction(T, None, fock, None, g, o, v, method="ccsdpt")

