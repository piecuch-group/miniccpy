import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar
from miniccpy.pspace import get_active_4h2p_pspace

def test_dipeom4a_ch2_sym():

    basis = '6-31g'
    nfrozen = 0

    geom = [["C", (0.0, 0.0, 0.0)],
            ["H", (0.0, 1.644403, -1.32213)],
            ["H", (0.0, -1.644403, -1.32213)]]

    fock, g, e_hf, o, v, orbsym = run_scf(geom, basis, nfrozen, symmetry="C2V", unit="Bohr", cartesian=False, charge=-2, return_orbsym=True)

    T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')
    H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

    nroot = 4
    R, omega_guess = run_guess(H1, H2, o, v, nroot, method="dipcis")

    omega = np.zeros(2)

    # Set up the list of 4h2p excitations corresponding to the active-space DIP-EOMCCSD(4h-2p){No} method
    # Here, we are using 10 active occupied orbitals, which corresponds to full DIP-EOMCCSD(4h-2p)
    no, nu = fock[o, v].shape

    # state = 0 is a B1-symmetric triplet
    r3_excitations = get_active_4h2p_pspace(no, nu, nacto=10, point_group="C2V", orbsym=orbsym, target_irrep="B1")
    R_b1, e_b1, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="dipeom4_p", state_index=[0], r3_excitations=r3_excitations)
    omega[0] = e_b1[0]

    # state = 1 is an A1-symmetric singlet
    r3_excitations = get_active_4h2p_pspace(no, nu, nacto=10, point_group="C2V", orbsym=orbsym, target_irrep="A1")
    R_a1, e_a1, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="dipeom4_p", state_index=[3], r3_excitations=r3_excitations)
    omega[1] = e_a1[0]

    #
    # Check the results
    #
    expected_vee = [-0.4700687744, -0.4490361545]
    for i, vee in enumerate(expected_vee):
        assert np.allclose(omega[i], vee, atol=1.0e-06)

if __name__ == "__main__":
    test_dipeom4a_ch2_sym()
