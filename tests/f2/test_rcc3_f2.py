import numpy as np
from miniccpy.driver import run_scf, run_cc_calc

def test_rcc3_f2():

        basis = 'cc-pvdz'
        nfrozen = 2
        # Define molecule geometry and basis set
        geom = [['F', (0.0, 0.0, -2.66816)],
                ['F', (0.0, 0.0,  2.66816)]]

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, unit="Bohr", rhf=True, cartesian=True, symmetry="D2H")

        T, E_corr = run_cc_calc(fock, g, o, v, method="rcc3")

        #
        # Check the results
        #
        assert np.allclose(-0.63843220, E_corr, atol=1.0e-08)

if __name__ == "__main__":
        test_rcc3_f2()