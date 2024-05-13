import numpy as np
from miniccpy.driver import run_scf, run_cc_calc

def test_ccsd_ch2():

    basis = '6-31g'
    nfrozen = 0

    geom = [['H', (0, 1.644403, 1.32213)],
            ['H', (0, -1.644403, 1.32213)],
            ['C', (0.0, 0.0, 0.0)]]

    fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen)

    T, E_corr = run_cc_calc(fock, g, o, v, method="ccsd")

if __name__ == "__main__":
    test_ccsd_ch2()
