from miniccpy.driver import run_scf, run_cc_calc, run_eomcc_calc, get_hbar, run_guess

basis = 'dz'
nfrozen = 1

# Define molecule geometry and basis set
geom = [['H', (0, 1.515263, -1.058898)], 
        ['H', (0, -1.515263, -1.058898)], 
        ['O', (0.0, 0.0, -0.0090)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen)

T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsdt')

H1, H2 = get_hbar(T, fock, g, o, v, method='ccsdt')

R, omega_guess = run_guess(H1, H2, o, v, 5, method="cis")
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomccsdt", state_index=[1, 2, 3], max_size=8)






