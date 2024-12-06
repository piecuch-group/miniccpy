import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, get_hbar, run_leftcc_calc, run_correction
from miniccpy.pspace import get_active_triples_pspace

def test_cct3_h8():

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

        # Set return_orbsym flag to return irreps of each MO in orbsym
        fock, g, e_hf, o, v, orbsym = run_scf(geom, basis, nfrozen, return_orbsym=True)

        no, nu = fock[o, v].shape

        ##########################################
        # Active-orbital-based CCSDt Calculation #
        ##########################################
        # Pick active space
        nacto = 2 # number of active occupied spinorbitals
        nactu = 2 # number of active unoccupied spinorbitals
        # num_active is the number of active occupied/unoccupied spinorbital indices constrainted to active set
        # num_active = 1 ==> |ijKAbc> [CCSDt(I), equivalent to CCSDt]
        #            = 2 ==> |iJKABc> [CCSDt(II)]
        #            = 3 ==> |IJKABC> [CCSDt(III)]
        num_active = 1 
        # Set up the list of point group symmetry-adapted triple excitations in the P space corresponding to CCSDt
        t3_excitations = get_active_triples_pspace(no, nu, nacto=nacto, nactu=nactu, point_group="D2H", orbsym=orbsym, target_irrep="AG", num_active=num_active)
        # Run CCSDt calculation via generic CC(P) code
        T, E_corr = run_cc_calc(fock, g, o, v, method='ccsdt_p', t3_excitations=t3_excitations)

        #######################################################
        # CC(t;3) Correction using the two-body approximation #
        #######################################################
        # Obtaine CCSD-like HBar
        H1, H2 = get_hbar((T[0], T[1]), fock, g, o, v, method="ccsd")
        # Solve left-CCSD-like equations
        L = run_leftcc_calc((T[0], T[1]), fock, H1, H2, o, v, method="left_ccsd") 
        # Run CC(t;3) / 2BA correction for triples missing from CCSDt
        E_correction = run_correction((T[0], T[1]), L, fock, H1, H2, o, v, method="cct3", nacto=nacto, nactu=nactu, num_active=num_active)

        assert np.allclose(E_corr, -0.17039345, atol=1.0e-07, rtol=1.0e-07)
        assert np.allclose(E_correction["A"], -0.0009284231, atol=1.0e-07, rtol=1.0e-07)
        assert np.allclose(E_correction["B"], -0.0008786944, atol=1.0e-07, rtol=1.0e-07)
        assert np.allclose(E_correction["C"], -0.0011658158, atol=1.0e-07, rtol=1.0e-07)
        assert np.allclose(E_correction["D"], -0.0011599478, atol=1.0e-07, rtol=1.0e-07)

if __name__ == "__main__":
        test_cct3_h8()
