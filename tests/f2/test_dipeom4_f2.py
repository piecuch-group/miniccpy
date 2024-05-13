import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar
from miniccpy.pspace import get_active_4h2p_pspace

def test_dipeom4_f2():

    basis = '6-31g'
    nfrozen = 2
    # Define molecule geometry and basis set
    geom = [['F', (0.0, 0.0, -2.66816)],
            ['F', (0.0, 0.0,  2.66816)]]

    fock, g, e_hf, o, v, orbsym = run_scf(geom, basis, nfrozen, unit="Bohr", symmetry="D2H", cartesian=False, charge=-2, return_orbsym=True)

    T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')
    H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

    nroot = 4
    R, omega_guess = run_guess(H1, H2, o, v, nroot, method="dipcis")

    no, nu = fock[o, v].shape
    r3_excitations = get_active_4h2p_pspace(no, nu, nacto=16, orbsym=orbsym, point_group="D2H", target_irrep="B1G")
    R_singlet, omega_triplet, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="dipeom4_p", state_index=[0], r3_excitations=r3_excitations, max_size=80)

    r3_excitations = get_active_4h2p_pspace(no, nu, nacto=16, orbsym=orbsym, point_group="D2H", target_irrep="AU")
    R_singlet, omega_singlet, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="dipeom4_p", state_index=[3], r3_excitations=r3_excitations, max_size=80)

    omega = np.array([omega_triplet[0], omega_singlet[0]])

    #
    # Check the results
    #
    expected_vee = [-0.133359091568, -0.133336529768]
    for i, vee in enumerate(expected_vee):
        assert np.allclose(omega[i], vee, atol=1.0e-07)

if __name__ == "__main__":
    test_dipeom4_f2()