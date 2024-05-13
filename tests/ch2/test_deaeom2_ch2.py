import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

def test_deaeom2_ch2():

    basis = '6-31g'
    nfrozen = 0

    geom = [["C", (0.0, 0.0, 0.0)],
            ["H", (0.0, 1.644403, -1.32213)],
            ["H", (0.0, -1.644403, -1.32213)]]

    fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, cartesian=True, charge=2)

    T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')
    H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

    no, nu = fock[o, v].shape
    R, omega_guess = run_guess(H1, H2, o, v, 5, method="deacis", mult=-1, nactu=nu)
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="deaeom2", state_index=[0, 1, 2, 3, 4])

    #
    # Check the results
    #
    assert np.allclose(Ecorr, -0.063555239304, atol=1.0e-07)

    assert np.allclose(omega[0], -1.061558167274, atol=1.0e-07)
    assert np.allclose(omega[1], -1.059279189171, atol=1.0e-07)
    assert np.allclose(omega[2], -1.059279189171, atol=1.0e-07)
    assert np.allclose(omega[3], -1.059279189171, atol=1.0e-07)
    assert np.allclose(omega[4], -0.992050058228, atol=1.0e-07)

if __name__ == "__main__":
    test_deaeom2_ch2()
