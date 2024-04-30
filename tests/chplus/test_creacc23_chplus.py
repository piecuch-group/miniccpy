import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_lefteomcc_calc, run_eom_correction

basis = '6-31g'
nfrozen = 0

# Define molecule geometry and basis set
geom = [['C', (0.0, 0.0, 2.1773/2)], 
        ['H', (0.0, 0.0, -2.1773/2)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, charge=1, unit="Bohr", symmetry="C2V")

T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')

H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

R, omega_guess = run_guess(H1, H2, o, v, 10, method="eacis")
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eaeom2", state_index=[3, 4], max_size=40)
L, omega_left = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eaeom2')

for i in range(len(R)):
    delta_T = run_eom_correction(T, R[i], L[i], r0[i], omega[i], fock, H1, H2, o, v, method="creacc23")

##### Check against CCpy - everything works but the D correction...

# from ccpy.drivers.driver import Driver
# from pyscf import gto, scf
#
# mol = gto.M(atom=geom, basis=basis, spin=0, symmetry="C2V", unit="Bohr", charge=1)
# mf = scf.RHF(mol)
# mf.kernel()
#
# driver = Driver.from_pyscf(mf, nfrozen=0)
# driver.run_cc(method="ccsd")
# driver.run_hbar(method="ccsd")
# driver.run_guess(method="eacisd", nact_occupied=0, nact_unoccupied=0, multiplicity=-1,
#                  roots_per_irrep={"B1": 6}, use_symmetry=False)
# driver.run_eaeomcc(method="eaeom2", state_index=[0, 2])
# driver.run_lefteaeomcc(method="left_eaeom2", state_index=[0, 2])
# driver.run_eaccp3(method="creacc23", state_index=[0, 2])
