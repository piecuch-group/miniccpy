"""
Compares the result of CC3 against the result obtained from Psi4
for the lowest-lying singlet state of the H2O molecule at the Re and
2Re structures, obtained from JCP 104, 8007 (1996).
"""
import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, get_hbar, run_leftcc_calc, run_correction, run_eomcc_calc, run_lefteomcc_calc, run_eom_correction, run_guess

from ccpy.drivers.driver import Driver
from pyscf import gto, scf

basis = '6-31g'
nfrozen = 0

re = 2

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

T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')
H1, H2 = get_hbar(T, fock, g, o, v, method="ccsd")
L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_ccsd")
delta_T = run_correction(T, L, H1, H2, o, v, method="crcc23")

R0, omega0 = run_guess(H1, H2, o, v, 5, method="cis", mult=1)
R, omega, r0 = run_eomcc_calc(R0, omega0, T, H1, H2, o, v, method='eomccsd', state_index=[0], maxit=200)
L, omega_left = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eomccsd', maxit=200)

for i in range(len(R)):
    print("Left = ", omega_left[i], "Right = ", omega[i])
    delta_T = run_eom_correction(T, R[i], L[i], r0[i], omega[i], H1, H2, o, v, method="creomcc23")

### CCpy test ###
mol = gto.M(atom=geom, symmetry="C2V", basis=basis, unit="Bohr")
mf = scf.RHF(mol)
mf.kernel()

driver = Driver.from_pyscf(mf, nfrozen=0)
driver.run_cc(method="ccsd")
driver.run_hbar(method="ccsd")
driver.run_guess(method="cis", multiplicity=1, roots_per_irrep={"B1": 1})
driver.run_eomcc(method="eomccsd", state_index=[1])
driver.run_leftcc(method="left_ccsd", state_index=[0, 1])
driver.run_ccp3(method="crcc23", state_index=[0, 1])
