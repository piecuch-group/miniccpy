# set allow numpy built with MKL to consume more threads for tensordot
import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

basis = '6-31g'
nfrozen = 0

re = 1

# Define molecule geometry and basis set
if re == 1:
    geom = [['H', (0, 1.515263, -1.058898)], 
            ['H', (0, -1.515263, -1.058898)], 
            ['O', (0.0, 0.0, -0.0090)]]
    target_state = 3
    target_vee = 0.3035583130
elif re == 2:
    geom = [["O", (0.0, 0.0, -0.0180)],
            ["H", (0.0, 3.030526, -2.117796)],
            ["H", (0.0, -3.030526, -2.117796)]]
    target_state = 12
    target_vee = 0.0282255024

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen)

T, Ecorr  = run_cc_calc(fock, g, o, v, method='cc3')
H1, H2 = get_hbar(T, fock, g, o, v, method='ccsdt')
R, omega_guess = run_guess(H1, H2, o, v, 15, method="cis")
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomcc3", state_index=[target_state], fock=fock, g=g, max_size=40)

print("Expected VEE = ", target_vee)
assert np.allclose(omega, target_vee, atol=1.0e-07)






