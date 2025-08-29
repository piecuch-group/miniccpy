import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, get_hbar, run_leftcc_calc
from miniccpy.ccsd_density import build_rdm1, build_rdm2
from miniccpy.energy import cc_energy_from_rdm

def test_ccsd_f2():

        basis = 'cc-pvdz'
        nfrozen = 2

        geom = [['F', (0.0, 0.0,  2.66816)],
                ['F', (0.0, 0.0, -2.66816)]]

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, cartesian=True, unit="Bohr", symmetry="D2H")

        T, E_corr = run_cc_calc(fock, g, o, v, method='ccsd')
        H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')
        L = run_leftcc_calc(T, fock, H1, H2, o, v, method='left_ccsd')

        # Construct 1- and 2-body RDMs
        rdm1 = build_rdm1(T, L)
        rdm2 = build_rdm2(T, L)
        E_corr_from_rdm = cc_energy_from_rdm(rdm1, rdm2, fock, g, o, v)

        #
        # Check the results
        #
        assert np.allclose(E_corr, -0.592466290032, atol=1.0e-07)
        assert np.allclose(E_corr_from_rdm, -0.592466290032, atol=1.0e-07)

if __name__ == "__main__":
        test_ccsd_f2()





