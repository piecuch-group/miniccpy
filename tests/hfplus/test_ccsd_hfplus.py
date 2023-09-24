import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_correction

basis = '6-31g'
nfrozen = 0
# Define molecule geometry and basis set
geom = [['H', (0.0, 0.0, -0.8)], 
        ['F', (0.0, 0.0,  0.8)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, unit="Angstrom", symmetry="C2V", charge=1, multiplicity=2)

T, E_corr = run_cc_calc(fock, g, o, v, method="ccsd")
e_correction = run_correction(T, fock, g, o, v, method="ccsdpt")
print("CCSD(T) total energy = ", E_corr + e_correction + e_hf)

#T, E_corr = run_cc_calc(fock, g, o, v, method="cc3-full")
#H1, H2 = get_hbar(T, fock, g, o, v, method="ccsdt")
#R, omega_guess = run_guess(H1, H2, o, v, 30, method="cis", mult=2)
#R, omega, r0 = run_eomcc_calc(R, omega_guess, (T[0],T[1]), H1, H2, o, v, method="eomccsd", state_index=[i for i in range(20)])








