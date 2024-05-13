import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, get_hbar, run_leftcc_calc, run_correction, run_eomcc_calc, run_lefteomcc_calc, run_eom_correction, run_guess

def test_creomcc23_h2o():

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

    fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen)

    T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')
    H1, H2 = get_hbar(T, fock, g, o, v, method="ccsd")
    L0 = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_ccsd")

    R0, omega0 = run_guess(H1, H2, o, v, 5, method="cis", mult=1)
    R, omega, r0 = run_eomcc_calc(R0, omega0, T, H1, H2, o, v, method='eomccsd', state_index=[0], maxit=200)
    L, omega_left = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eomccsd', maxit=200)

    delta_T = []
    delta_T.append(run_correction(T, L0, fock, H1, H2, o, v, method="crcc23"))
    for i in range(len(R)):
        delta_T.append(run_eom_correction(T, R[i], L[i], r0[i], omega[i], fock, H1, H2, o, v, method="creomcc23"))

    #
    # Check the results
    #
    assert np.allclose(Ecorr, -0.291219152750, atol=1.0e-07)
    assert np.allclose(delta_T[0]["A"], -0.009907050495912655, atol=1.0e-07)
    assert np.allclose(delta_T[0]["D"], -0.01333695816624863, atol=1.0e-07)
    assert np.allclose(delta_T[1]["A"], -0.011509068318432032, atol=1.0e-07)
    assert np.allclose(delta_T[1]["D"], -0.016591168450218817, atol=1.0e-07)

if __name__ == "__main__":
    test_creomcc23_h2o()