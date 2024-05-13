import numpy as np
from miniccpy.driver import run_scf, run_cc_calc

def test_rcc3_hf():

        basis = '6-31g'
        nfrozen = 0
        # Define molecule geometry and basis set
        geom = [['H', (0.0, 0.0, -0.8)],
                ['F', (0.0, 0.0,  0.8)]]

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, unit="Angstrom", symmetry="C2V", rhf=True)

        T, E_corr = run_cc_calc(fock, g, o, v, method="rcc3")

        #
        # Check the results
        #
        assert np.isclose(E_corr, -0.178932834216145, atol=1e-7)

if __name__ == "__main__":
        test_rcc3_hf()





