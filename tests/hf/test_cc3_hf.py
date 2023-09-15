import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

basis = '6-31g'
nfrozen = 0
# Define molecule geometry and basis set
geom = [['H', (0.0, 0.0, -0.8)], 
        ['F', (0.0, 0.0,  0.8)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, unit="Angstrom", symmetry="C2V")

T, E_corr = run_cc_calc(fock, g, o, v, method="cc3")
assert np.isclose(E_corr, -0.178932834216145, atol=1e-9)

H1, H2 = get_hbar(T, fock, g, o, v, method="ccsdt")
R, omega_guess = run_guess(H1, H2, o, v, 10, method="cis", mult=1)
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomcc3", state_index=[0, 1], fock=fock, g=g)






