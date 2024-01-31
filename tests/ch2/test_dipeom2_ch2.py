import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

basis = '6-31g'
nfrozen = 0

geom = [["C", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.644403, -1.32213)],
        ["H", (0.0, -1.644403, -1.32213)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, cartesian=True, charge=-2)

T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')
H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

nroot = 1
R, omega_guess = run_guess(H1, H2, o, v, nroot, method="dipcis")
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="dipeom2", state_index=[0])





