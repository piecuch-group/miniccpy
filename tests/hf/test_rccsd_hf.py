import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, get_hbar, run_leftcc_calc

def test_rccsd_hf():

        basis = 'cc-pvdz'
        nfrozen = 0

        geom = [['H', (0.0, 0.0, -1.0)],
                ['F', (0.0, 0.0,  1.0)]]

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200, unit="Angstrom", rhf=True, cartesian=False)

        T, E_corr = run_cc_calc(fock, g, o, v, method='rccsd', maxit=80)
        H1, H2 = get_hbar(T, fock, g, o, v, method="rccsd")
        L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_rccsd")

        #
        # Check the results
        #
        assert np.allclose(-0.277969253916, E_corr, atol=1.0e-08)

if __name__ == "__main__":
        test_rccsd_hf()







