import numpy as np
from miniccpy.driver import run_scf_gamess, run_cc_calc, get_hbar, run_leftcc_calc


fock, g, e_hf, o, v = run_scf_gamess("f2.FCIDUMP", 18, 40, 2, rhf=True)

T, E_corr = run_cc_calc(fock, g, o, v, method='rccsd', maxit=80, convergence=1.0e-09)
H1, H2 = get_hbar(T, fock, g, o, v, method="rccsd")
L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_rccsd", convergence=1.0e-09)

nu, no = T[0].shape

#for a in range(nu):
#    for i in range(no):
#        print(L[0][a, i])

#for i in range(no):
#    for a in range(nu):
#        for j in range(no):
#            for b in range(nu):
#                print(L[1][a, b, i, j])
