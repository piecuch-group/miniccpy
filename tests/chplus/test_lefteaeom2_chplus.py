from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_lefteomcc_calc

basis = '6-31g'
nfrozen = 0

# Define molecule geometry and basis set
geom = [['C', (0.0, 0.0, 2.1773/2)], 
        ['H', (0.0, 0.0, -2.1773/2)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, charge=1, unit="Bohr", symmetry="C2V")

T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')

H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

R, omega_guess = run_guess(H1, H2, o, v, 10, method="eacis")
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eaeom2", state_index=[0, 4], max_size=40)
L, omega_left = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eaeom2')





