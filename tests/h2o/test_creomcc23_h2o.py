import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, get_hbar, run_leftcc_calc, run_correction, run_eomcc_calc, run_lefteomcc_calc, run_eom_correction, run_guess

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
H1, H2 = get_hbar(T, fock, g, o, v, method="ccsd")
L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_ccsd")
delta_T = run_correction(T, L, fock, H1, H2, o, v, method="crcc23")

R0, omega0 = run_guess(H1, H2, o, v, 5, method="cis", mult=1)
R, omega, r0 = run_eomcc_calc(R0, omega0, T, H1, H2, o, v, method='eomccsd', state_index=[0], maxit=200)
L, omega_left = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eomccsd', maxit=200)

for i in range(len(R)):
    delta_T = run_eom_correction(T, R[i], L[i], r0[i], omega[i], fock, H1, H2, o, v, method="creomcc23")
