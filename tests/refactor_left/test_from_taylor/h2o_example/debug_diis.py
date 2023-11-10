import numpy as np
from miniccpy.driver import run_scf_gamess, run_cc_calc, get_hbar, run_leftcc_calc


fock, g, e_hf, o, v = run_scf_gamess("h2o.FCIDUMP", 20, 50, 0, rhf=True)

T, E_corr = run_cc_calc(fock, g, o, v, method='rccsd', maxit=80, convergence=1.0e-08)
H1, H2 = get_hbar(T, fock, g, o, v, method="rccsd")
L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_rccsd", convergence=1.0e-08)

