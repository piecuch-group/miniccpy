import numpy as np
from miniccpy.driver import run_scf, run_cc_calc

def test_rccd_hf():

        basis = '6-31g'
        nfrozen = 0

        geom = [['H', (0.0, 0.0, -0.8)],
                ['F', (0.0, 0.0,  0.8)]]

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200, unit="Angstrom", rhf=True)

        T, E_corr = run_cc_calc(fock, g, o, v, method='rccd', maxit=80)

        #
        # Check the results
        #
        assert np.allclose(-0.1691379459, E_corr, atol=1.0e-08)

if __name__ == "__main__":
        test_rccd_hf()




