import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_lefteomcc_calc, run_eom_correction

def test_creacc23_chplus():

    basis = '6-31g'
    nfrozen = 0

    # Define molecule geometry and basis set
    geom = [['C', (0.0, 0.0, 2.1773/2)],
            ['H', (0.0, 0.0, -2.1773/2)]]

    fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, charge=1, unit="Bohr", symmetry="C2V")

    T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')

    H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

    R, omega_guess = run_guess(H1, H2, o, v, 10, method="eacis")
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eaeom2", state_index=[3, 4], max_size=40)
    L, omega_left = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eaeom2')

    delta_T = []
    for i in range(len(R)):
        delta_T.append(run_eom_correction(T, R[i], L[i], r0[i], omega[i], fock, H1, H2, o, v, method="creacc23"))

    #
    # Check the results
    #
    assert np.allclose(Ecorr, -0.072469559048, atol=1.0e-07)

    assert np.allclose(omega[0], -0.365169292472, atol=1.0e-07)
    assert np.allclose(delta_T[0]["A"], 0.0008705172844524277, atol=1.0e-07)
    assert np.allclose(delta_T[0]["D"], 0.0013667699807270368, atol=1.0e-07)

    assert np.allclose(omega[1], -0.073559341175, atol=1.0e-07)
    assert np.allclose(delta_T[1]["A"], 0.0007695056519347155, atol=1.0e-07)
    assert np.allclose(delta_T[1]["D"], 0.002214488970312919, atol=1.0e-07)

if __name__ == "__main__":
    test_creacc23_chplus()
