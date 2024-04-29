import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar
from miniccpy.pspace import get_active_4h2p_pspace

basis = '6-31g'
nfrozen = 0

geom = [["C", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.644403, -1.32213)],
        ["H", (0.0, -1.644403, -1.32213)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, symmetry="C2V", unit="Bohr", cartesian=False, charge=-2)

T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')
H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

nroot = 4
R, omega_guess = run_guess(H1, H2, o, v, nroot, method="dipcis")

# Set up the list of 4h2p excitations corresponding to the active-space DIP-EOMCCSD(4h-2p){No} method
no, nu = fock[o, v].shape
# Here, we are using 10 active occupied orbitals, which corresponds to full DIP-EOMCCSD(4h-2p)
r3_excitations = get_active_4h2p_pspace(no, nu, nacto=10)
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="dipeom4_p", state_index=[0, 3], r3_excitations=r3_excitations)

#
# Check the results
#
expected_vee = [-0.4700687744, -0.4490361545]
for i, vee in enumerate(expected_vee):
    assert np.allclose(omega[i], vee, atol=1.0e-06)
