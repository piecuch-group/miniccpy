import numpy as np
from miniccpy.driver import run_scf, run_cc_calc

def test_rccsdt_f2():

        basis = 'cc-pvdz'
        nfrozen = 2

        geom = [['F', (0.0, 0.0,  2.66816/2)],
                ['F', (0.0, 0.0, -2.66816/2)]]

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, cartesian=True, unit="Bohr", rhf=True, symmetry="D2H")

        T, E_corr = run_cc_calc(fock, g, o, v, method='rccsdt')

        #
        # Check the results
        #
        assert np.allclose(-0.416431388083, E_corr, rtol=1.0e-08)

if __name__ == "__main__":
        test_rccsdt_f2()


