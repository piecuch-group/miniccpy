import numpy as np
from miniccpy.driver import run_scf, run_cc_calc
from miniccpy.pspace import get_active_triples_pspace

def test_ccsdt1_h2o():

    basis = '6-31g'
    nfrozen = 0

    re = 2

    # Define molecule geometry and basis set
    if re == 1:
        geom = [['H', (0, 1.515263, -1.058898)],
                ['H', (0, -1.515263, -1.058898)],
                ['O', (0.0, 0.0, -0.0090)]]
    elif re == 2:
        geom = [["O", (0.0, 0.0, -0.0180)],
                ["H", (0.0, 3.030526, -2.117796)],
                ["H", (0.0, -3.030526, -2.117796)]]

    fock, g, e_hf, o, v, orbsym = run_scf(geom, basis, nfrozen, return_orbsym=True, symmetry="C2V")

    no, nu = fock[o, v].shape

    # Set up the list of triple excitations corresponding to the active-space CCSDt calculation
    t3_excitations = get_active_triples_pspace(no, nu, nacto=4, nactu=4, point_group="C2V", target_irrep="A1", orbsym=orbsym)
    T, E_corr = run_cc_calc(fock, g, o, v, method='ccsdt_p', t3_excitations=t3_excitations)

    #
    # Check the results
    #
    assert np.allclose(E_corr, -0.30334782, atol=1.0e-07)

if __name__ == "__main__":
    test_ccsdt1_h2o()