import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, get_hbar, run_leftcc_calc
from miniccpy.rccsd_density import build_rdm1, build_rdm2
from miniccpy.energy import cc_energy_from_rdm

basis = 'cc-pvdz'
nfrozen = 2

geom = [['F', (0.0, 0.0,  2.66816)],
        ['F', (0.0, 0.0, -2.66816)]] 

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, cartesian=True, unit="Bohr", rhf=True)

T, E_corr = run_cc_calc(fock, g, o, v, method='rccsd')
H1, H2 = get_hbar(T, fock, g, o, v, method='rccsd')
L = run_leftcc_calc(T, H1, H2, o, v, method='left_rccsd')

print("Norm of L1 = ", np.linalg.norm(L[0].flatten()))
print("Norm of L2 = ", np.linalg.norm(L[1].flatten()))

rdm1 = build_rdm1(T, L)
rdm2 = build_rdm2(T, L)
E_corr2 = cc_energy_from_rdm(rdm1, rdm2, fock, g, o, v)

print("Energy from RDM = ", E_corr2)
print("Energy from CC = ", E_corr)






