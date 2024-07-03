import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_lefteomcc_calc, run_dip_correction

def test_dipeom3_ch2():

    basis = '6-31g'
    nfrozen = 0

    geom = [["C", (0.0, 0.0, 0.0)],
            ["H", (0.0, 1.644403, -1.32213)],
            ["H", (0.0, -1.644403, -1.32213)]]

    fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, symmetry="C2V", unit="Bohr", cartesian=False, charge=-2)

    T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')
    H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

    nroot = 4
    R, omega_guess = run_guess(H1, H2, o, v, nroot, method="dipcis")
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="dipeom3", state_index=[0, 3])
    L, omega = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method="left_dipeom3")

    for i in range(len(R)):
        delta = run_dip_correction(T, R[i], R[i], omega[i], fock, g, H1, H2, o, v, method="dipeom34") 

    #
    # Check the results
    #
    expected_vee = [-0.4590911571, -0.4422414052]
    for i, vee in enumerate(expected_vee):
        assert np.allclose(omega[i], vee, atol=1.0e-06)

if __name__ == "__main__":
    test_dipeom3_ch2()
