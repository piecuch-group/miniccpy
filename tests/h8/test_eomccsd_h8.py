import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_eomcc_calc, get_hbar, run_guess

def test_eomccsd_h8():

        basis = 'dz'
        nfrozen = 0

        # Define molecule geometry and basis set
        geom = [['H', ( 2.4143135624,  1.000, 0.000)],
                ['H', (-2.4143135624,  1.000, 0.000)],
                ['H', ( 2.4143135624, -1.000, 0.000)],
                ['H', (-2.4143135624, -1.000, 0.000)],
                ['H', ( 1.000,  2.4142135624, 0.000)],
                ['H', (-1.000,  2.4142135624, 0.000)],
                ['H', ( 1.000, -2.4142135624, 0.000)],
                ['H', (-1.000, -2.4142135624, 0.000)],
                ]

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen)

        T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')

        H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

        R, omega_guess = run_guess(H1, H2, o, v, 20, method="cisd", mult=1, nacto=6, nactu=6)
        R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method='eomccsd', state_index=[0, 1, 2])

        #
        # Check the results
        #
        assert np.allclose(Ecorr, -0.155467646059, atol=1.0e-07)
        assert np.allclose(omega[0], 0.068320236535, atol=1.0e-07)
        assert np.allclose(omega[1], 0.069683076218, atol=1.0e-07)
        assert np.allclose(omega[2], 0.290176069263, atol=1.0e-07)

if __name__ == "__main__":
        test_eomccsd_h8()
