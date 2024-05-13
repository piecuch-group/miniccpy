import numpy as np
from miniccpy.driver import run_scf, run_cc_calc

def test_rccsd_h4():

        basis = 'cc-pvdz'
        nfrozen = 0
        Re = 1.0

        geom = [['H', (-Re, -Re, 0.000)],
                ['H', (-Re,  Re, 0.000)],
                ['H', (Re, -Re, 0.000)],
                ['H', (Re,  Re, 0.000)]]

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200, rhf=True)

        T, E_corr = run_cc_calc(fock, g, o, v, method='rccsd', maxit=80)

        #
        # Check the results
        #
        assert np.allclose(E_corr, -0.084308599849, atol=1.0e-07)

if __name__ == "__main__":
        test_rccsd_h4()