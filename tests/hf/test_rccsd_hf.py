import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, get_hbar, run_leftcc_calc

basis = 'cc-pvdz'
nfrozen = 0

geom = [['H', (0.0, 0.0, -1.0)], 
        ['F', (0.0, 0.0,  1.0)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200, unit="Angstrom", rhf=True, cartesian=False)

T, E_corr = run_cc_calc(fock, g, o, v, method='rccsd', maxit=80)
H1, H2 = get_hbar(T, fock, g, o, v, method="rccsd")
L = run_leftcc_calc(H1, H2, T, o, v, method="left_rccsd")

#print("Norm of L1 = ", np.linalg.norm(L[0].flatten()))
#print("Norm of L2 = ", np.linalg.norm(L[1].flatten()))

#print("H1")
#print("----")
#print("H1[o, v] = ", np.linalg.norm(H1[o, v].flatten()))
#print("H1[o, o] = ", np.linalg.norm(H1[o, o].flatten()))
#print("H1[v, v] = ", np.linalg.norm(H1[v, v].flatten()))
#print("H2")
#print("------")
#print("H2[o, o, o, o] = ", np.linalg.norm(H2[o, o, o, o].flatten()))
#print("H2[o, o, o, v] = ", np.linalg.norm(H2[o, o, o, v].flatten()))
#print("H2[o, o, v, o] = ", np.linalg.norm(H2[o, o, v, o].flatten()))
#print("H2[o, v, o, o] = ", np.linalg.norm(H2[o, v, o, o].flatten()))
#print("H2[v, o, o, o] = ", np.linalg.norm(H2[v, o, o, o].flatten()))
#print("H2[v, o, o, v] = ", np.linalg.norm(H2[v, o, o, v].flatten()))
#print("H2[v, o, v, o] = ", np.linalg.norm(H2[v, o, v, o].flatten()))
#print("H2[o, v, o, v] = ", np.linalg.norm(H2[o, v, o, v].flatten()))
#print("H2[o, v, v, o] = ", np.linalg.norm(H2[o, v, v, o].flatten()))
#print("H2[v, o, v, v] = ", np.linalg.norm(H2[v, o, v, v].flatten()))
#print("H2[o, v, v, v] = ", np.linalg.norm(H2[o, v, v, v].flatten()))
#print("H2[v, v, o, v] = ", np.linalg.norm(H2[v, v, o, v].flatten()))
#print("H2[v, v, v, o] = ", np.linalg.norm(H2[v, v, v, o].flatten()))
#print("H2[v, v, v, v] = ", np.linalg.norm(H2[v, v, v, v].flatten()))

assert np.allclose(-0.277969253916, E_corr, atol=1.0e-08)









