from pathlib import Path
import numpy as np
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_lefteomcc_calc, run_eom_correction

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/chplus.FCIDUMP", 6, 26, 0)

T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')

H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

R0, omega0 = run_guess(H1, H2, o, v, 100, method="cis")

R, omega, r0 = run_eomcc_calc(R0, omega0, T, H1, H2, o, v, method='eomccsd', state_index=[17], maxit=200)
L, omega_left = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eomccsd', maxit=200)

for i in range(len(R)):
    delta_T = run_eom_correction(T, R[i], L[i], r0[i], omega[i], fock, H1, H2, o, v, method="creomcc23")
    
    print(f"CR-EOMCC(2,3) for Root {i}")
    print("--------------------------")
    print("EOMCCSD = ", e_hf + Ecorr + omega[i])
    print("CR-EOMCC(2,3)_A = ", e_hf + Ecorr + omega[i] + delta_T["A"])
    print("CR-EOMCC(2,3)_B = ", e_hf + Ecorr + omega[i] + delta_T["B"])
    print("CR-EOMCC(2,3)_C = ", e_hf + Ecorr + omega[i] + delta_T["C"])
    print("CR-EOMCC(2,3)_D = ", e_hf + Ecorr + omega[i] + delta_T["D"])
    print("")






