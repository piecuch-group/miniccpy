import numpy as np
from miniccpy.driver import run_scf, run_cc_calc
from miniccpy.pspace import get_active_triples_pspace

def test_ccsdt1_h8():

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

        fock, g, e_hf, o, v, orbsym = run_scf(geom, basis, nfrozen, return_orbsym=True)

        no, nu = fock[o, v].shape

        # Set up the list of triple excitations in the P space corresponding to CCSDt
        t3_excitations = get_active_triples_pspace(no, nu, nacto=6, nactu=6, point_group="D2H", orbsym=orbsym, target_irrep="AG")
        T, E_corr = run_cc_calc(fock, g, o, v, method='ccsdt_p', t3_excitations=t3_excitations)

        assert np.allclose(E_corr, -0.17346159, atol=1.0e-07)

if __name__ == "__main__":
        test_ccsdt1_h8()