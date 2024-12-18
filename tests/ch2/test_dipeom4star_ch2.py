import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_dip_correction

def test_dipeom4star_ch2():

    basis = '6-31g'
    nfrozen = 0

    geom = [["C", (0.0, 0.0, 0.0)],
            ["H", (0.0, 1.644403, -1.32213)],
            ["H", (0.0, -1.644403, -1.32213)]]

    fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, symmetry="C2V", unit="Bohr", cartesian=False, charge=-2)

    T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')
    T, H = get_hbar(T, fock, g, o, v, method='ccsdta')
    H1, H2 = H

    nroot = 4
    R, omega_guess = run_guess(H1, H2, o, v, nroot, method="dipcis")
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="dipeom3", state_index=[0, 3])
    delta_star = []
    for i in range(len(R)):
        delta_star.append(run_dip_correction(T, R[i], None, omega[i], fock, g, H1, H2, o, v, method="dipeom4_star"))

    expected_vee = [-0.45663637, -0.44009454]
    expected_correction = [-0.0217160701, -0.0221398782]
    #
    # Check the results
    #
    assert np.allclose(Ecorr, -0.10364237)
    for i in range(len(expected_vee)):
        assert np.allclose(omega[i], expected_vee[i])
        assert np.allclose(delta_star[i]["A"], expected_correction[i])




if __name__ == "__main__":
    test_dipeom4star_ch2()
