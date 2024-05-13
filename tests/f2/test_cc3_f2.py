import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, get_hbar, run_leftcc_calc

def test_cc3_f2():

        basis = '6-31g'
        nfrozen = 2
        # Define molecule geometry and basis set
        geom = [['F', (0.0, 0.0, -2.66816)],
                ['F', (0.0, 0.0,  2.66816)]]

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, unit="Bohr", symmetry="D2H")

        T, E_corr = run_cc_calc(fock, g, o, v, method="cc3", out_of_core=False)

        H1, H2 = get_hbar(T, fock, g, o, v, method='cc3')

        L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_cc3", g=g, diis_size=10)

        #
        # Check the results
        #
        assert np.allclose(-0.499092163794, E_corr, atol=1.0e-08)

if __name__ == "__main__":
        test_cc3_f2()