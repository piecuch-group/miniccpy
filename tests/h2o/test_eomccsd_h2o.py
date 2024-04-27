from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

basis = '6-31g'
nfrozen = 0

re = 1

# Define molecule geometry and basis set
if re == 1:
    geom = [['H', (0, 1.515263, -1.058898)],
            ['H', (0, -1.515263, -1.058898)],
            ['O', (0.0, 0.0, -0.0090)]]
elif re == 2:
    geom = [["O", (0.0, 0.0, -0.0180)],
            ["H", (0.0, 3.030526, -2.117796)],
            ["H", (0.0, -3.030526, -2.117796)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen)

T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')

H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

R, omega_guess = run_guess(H1, H2, o, v, 10, method="cis")
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomccsd", state_index=[12], max_size=20)






