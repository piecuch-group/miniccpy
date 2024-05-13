import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

def test_eomccsd_hf():

        basis = '6-31g'
        nfrozen = 0
        # Define molecule geometry and basis set
        geom = [['H', (0.0, 0.0, -0.8)],
                ['F', (0.0, 0.0,  0.8)]]

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, unit="Angstrom", symmetry="C2V")

        T, E_corr = run_cc_calc(fock, g, o, v, method="ccsd")

        H1, H2 = get_hbar(T, fock, g, o, v, method="ccsd")

        R, omega_guess = run_guess(H1, H2, o, v, 50, method="cisd", mult=1, nacto=6, nactu=6)
        R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomccsd", state_index=[0, 2, 3, 4], max_size=20)

        #
        # Check the results
        #
        assert np.allclose(E_corr, -0.175767067992, atol=1.0e-07)
        assert np.allclose(omega[0], 0.104200793902, atol=1.0e-07)
        assert np.allclose(omega[1], 0.343117586448, atol=1.0e-07)
        assert np.allclose(omega[2], 0.695261072467, atol=1.0e-07)
        assert np.allclose(omega[3], 0.704009655714, atol=1.0e-07)

if __name__ == "__main__":
        test_eomccsd_hf()
