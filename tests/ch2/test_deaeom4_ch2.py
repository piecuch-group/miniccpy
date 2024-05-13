import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

def test_deaeom4_ch2():

    basis = '6-31g'
    nfrozen = 0

    geom = [["C", (0.0, 0.0, 0.0)],
            ["H", (0.0, 1.644403, -1.32213)],
            ["H", (0.0, -1.644403, -1.32213)]]

    fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, cartesian=True, charge=2)

    T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')
    H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')
    R, omega_guess = run_guess(H1, H2, o, v, 10, method="deacis", mult=-1, nactu=10)

    state_index = [0, 1, 4, 5, 6]
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="deaeom4", state_index=state_index, convergence=1.0e-08, maxit=100, out_of_core=True)

    expected_vee = [-1.2063288624, -1.2280321410, -1.1434807780, -1.0428117510, -0.9119057911]

    #
    # Check the results
    #
    for i, vee in enumerate(expected_vee):
        assert np.allclose(omega[i], vee, atol=1.0e-07)

if __name__ == "__main__":
    test_deaeom4_ch2()
