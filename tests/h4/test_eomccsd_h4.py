import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_lefteomcc_calc

def test_eomccsd_h4():

        basis = '6-31g'
        nfrozen = 0

        # Define molecule geometry and basis set
        geom = [['H', (-2.000, -2.000, 0.000)],
                ['H', (-2.000,  2.000, 0.000)],
                ['H', ( 2.000, -2.000, 0.000)],
                ['H', ( 2.000,  2.000, 0.000)]]

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen)

        T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')

        H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

        R, omega_guess = run_guess(H1, H2, o, v, 10, method="cis")
        R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method='eomccsd', state_index=[0, 1, 2, 3, 4, 5, 6])
        L, omega_left = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eomccsd')

        #
        # Check the results
        #
        assert np.allclose(Ecorr, -0.202702610372, atol=1.0e-07)
        assert np.allclose(omega[0], -0.033276262135, atol=1.0e-07)
        assert np.allclose(omega[1], -0.033276262135, atol=1.0e-07)
        assert np.allclose(omega[2], -0.033276262135, atol=1.0e-07)
        assert np.allclose(omega[3], -0.035215139069, atol=1.0e-07)
        assert np.allclose(omega[4], -0.035215139069, atol=1.0e-07)
        assert np.allclose(omega[5], -0.035215139069, atol=1.0e-07)
        assert np.allclose(omega[6], -0.069539030869, atol=1.0e-07)

if __name__ == "__main__":
        test_eomccsd_h4()