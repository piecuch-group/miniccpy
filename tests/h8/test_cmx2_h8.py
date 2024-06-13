import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_leftcc_calc, get_hbar, run_cmx_calc

def test_cmx2_h8():

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

        T, E_corr = run_cc_calc(fock, g, o, v, method='ccsd')
        H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')
        L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_ccsd")
        cmx_corr = run_cmx_calc(T, L, E_corr, H1, H2, o, v, method="cmx2_ccsd")

        #
        # Check the results
        #
        #assert np.allclose(E_corr, -0.155467646059, atol=1.0e-07)

if __name__ == "__main__":
        test_cmx2_h8()
