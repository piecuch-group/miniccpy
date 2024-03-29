from miniccpy.driver import run_scf, run_cc_calc, get_hbar, run_leftcc_calc
from miniccpy.rccsd_density import build_rdm1, build_rdm2
from miniccpy.energy import cc_energy_from_rdm

basis = 'dz'
nfrozen = 0

# Define molecule geometry and basis set
geom = [['H', ( 2.4143135624,  1.000, 0.000)], 
        ['H', (-2.4143135624,  1.000, 0.000)],
        ['H', ( 2.4143135624, -1.000, 0.000)],
        ['H', (-2.4143135624, -1.000, 0.000)],
        ['H', ( 1.000,  2.4142135624, 0.000)],
        ['H', (-1.000,  2.4142135624, 0.000)],
        ['H', ( 1.000, -2.4142135624, 0.000)],
        ['H', (-1.000, -2.4142135624, 0.000)],
        ]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, rhf=True)

T, E_corr = run_cc_calc(fock, g, o, v, method='rccsd')
H1, H2 = get_hbar(T, fock, g, o, v, method="rccsd")
L = run_leftcc_calc(T, H1, H2, o, v, method="left_rccsd")

rdm1 = build_rdm1(T, L)
rdm2 = build_rdm2(T, L)
E_corr2 = cc_energy_from_rdm(rdm1, rdm2, fock, g, o, v)

print("CC energy = ", E_corr)
print("CC energy from RDMs = ", E_corr2)






