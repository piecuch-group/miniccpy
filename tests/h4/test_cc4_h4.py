import numpy as np
from miniccpy.driver import run_scf, run_cc_calc

def test_cc4_h4():

        basis = 'dz'
        nfrozen = 0
        Re = 1.0

        geom = [['H', (-Re, -Re, 0.000)],
                ['H', (-Re,  Re, 0.000)],
                ['H', (Re, -Re, 0.000)],
                ['H', (Re,  Re, 0.000)]]

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200)

        T, E_corr = run_cc_calc(fock, g, o, v, method='cc4', maxit=80)

        print("CC4 correlation energy:", E_corr)
        print("CCSDTQ correlation energy:", -0.064295914558) 

if __name__ == "__main__":
        test_cc4_h4()
