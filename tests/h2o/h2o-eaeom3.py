import numpy as np
# set allow numpy built with MKL to consume more threads for tensordot
import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

basis = 'dz'
nfrozen = 0

# Define molecule geometry and basis set
geom = [['H', (0, 1.515263, -1.058898)], 
        ['H', (0, -1.515263, -1.058898)], 
        ['O', (0.0, 0.0, -0.0090)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen)

T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')

H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

oa = slice(0, 10, 2)
ob = slice(1, 10, 2)
va = slice(0, 28, 2)
vb = slice(1, 28, 2)

print("H(voov)")
h2_voov = H2[v, o, o, v]
h2_ovvo = np.transpose(H2[v, o, o, v], (1, 0, 3, 2))
h2_vovo = -np.transpose(H2[v, o, o, v], (0, 1, 3, 2))
h2_ovov = -np.transpose(H2[v, o, o, v], (1, 0, 2, 3))
print("H.aa.voov = ", np.linalg.norm(h2_voov[va, oa, oa, va].flatten()))
print("H.ab.voov = ", np.linalg.norm(h2_voov[va, ob, oa, vb].flatten()))
print("H.ab.ovvo = ", np.linalg.norm(h2_ovvo[oa, vb, va, ob].flatten()))
print("H.ab.vovo = ", np.linalg.norm(h2_vovo[va, ob, va, ob].flatten()))
print("H.ab.ovov = ", np.linalg.norm(h2_ovov[oa, vb, oa, vb].flatten()))
print("H.bb.voov = ", np.linalg.norm(h2_voov[vb, ob, ob, vb].flatten()))

R, omega_guess = run_guess(H1, H2, o, v, 10, method="eacis")
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eaeom3", state_index=[1, 5], max_size=30, convergence=1.0e-10)






