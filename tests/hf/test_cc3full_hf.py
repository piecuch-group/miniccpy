import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_lefteomcc_calc, run_leftcc_calc

def test_cc3full_hf():

        basis = '6-31g'
        nfrozen = 0
        # Define molecule geometry and basis set
        geom = [['H', (0.0, 0.0, -0.8)],
                ['F', (0.0, 0.0,  0.8)]]

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, unit="Angstrom", symmetry="C2V")

        T, E_corr = run_cc_calc(fock, g, o, v, method="cc3-full")

        H1, H2 = get_hbar(T, fock, g, o, v, method="ccsdt")
        L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_cc3-full", g=g)

        R, omega_guess = run_guess(H1, H2, o, v, 30, method="cisd", mult=1, nacto=6, nactu=6)
        R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomcc3-lin", state_index=[0], fock=fock, g=g)
        L, omega = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method="left_eomcc3-lin", fock=fock, g=g)

        #
        # Check the results
        #
        assert np.allclose(E_corr, -0.178932834216145, atol=1.0e-07)
        assert np.allclose(omega[0], 0.100838711471, atol=1.0e-07)

if __name__ == "__main__":
        test_cc3full_hf()




