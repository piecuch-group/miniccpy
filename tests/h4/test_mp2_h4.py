import numpy as np
from miniccpy.driver import run_scf, run_mpn_calc

def test_mp2_h4():

        basis = 'cc-pvdz'
        nfrozen = 0
        Re = 1.0

        geom = [['H', (-Re, -Re, 0.000)],
                ['H', (-Re,  Re, 0.000)],
                ['H', (Re, -Re, 0.000)],
                ['H', (Re,  Re, 0.000)]]

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200)

        E_corr = run_mpn_calc(fock, g, o, v, method='mp2')

        #
        # Check the results
        #
        assert np.allclose(E_corr, -0.065015524507, atol=1.0e-07)

if __name__ == "__main__":
        test_mp2_h4()
