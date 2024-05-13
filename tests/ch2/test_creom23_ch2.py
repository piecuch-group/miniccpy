import numpy as np
from pathlib import Path
from miniccpy.driver import (run_scf_gamess, 
                             run_cc_calc, run_leftcc_calc,
                             get_hbar,
                             run_guess, run_eomcc_calc, run_lefteomcc_calc, 
                             run_correction, run_eom_correction)

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_creom23_ch2():

    deltaT = []

    fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/ch2-avdz-koch.FCIDUMP", 8, 27, nfrozen=0)

    T, Ecorr = run_cc_calc(fock, g, o, v, method="ccsd")

    H1, H2 = get_hbar(T, fock, g, o, v, method="ccsd")

    L = run_leftcc_calc(T, fock, H1, H2, o, v, method='left_ccsd')
    deltaT.append(run_correction(T, L, fock, H1, H2, o, v, method="crcc23"))

    R, omega_guess = run_guess(H1, H2, o, v, 20, method="cisd", mult=1, nacto=2, nactu=4)
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomccsd", state_index=[0, 1, 2, 3, 4])
    L, omega_left = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eomccsd')

    for i in range(len(R)):
        deltaT.append(run_eom_correction(T, R[i], L[i], r0[i], omega[i], fock, H1, H2, o, v, method="creomcc23"))

    #
    # Check the results
    #
    assert np.allclose(Ecorr, -0.140400623438, atol=1.0e-07)
    assert np.allclose(deltaT[0]["A"], -0.002715143792878042, atol=1.0e-07)
    assert np.allclose(deltaT[0]["D"], -0.0035640422585983076, atol=1.0e-07)

    assert np.allclose(omega[0], 0.065435318461, atol=1.0e-07)
    assert np.allclose(deltaT[1]["A"], -0.0016859307612009068, atol=1.0e-07)
    assert np.allclose(deltaT[1]["D"], -0.00223583254851049, atol=1.0e-07)

    assert np.allclose(omega[1], 0.224588637726, atol=1.0e-07)
    assert np.allclose(deltaT[2]["A"], -0.036090027969107825, atol=1.0e-07)
    assert np.allclose(deltaT[2]["D"], -0.05476465056846732, atol=1.0e-07)

    assert np.allclose(omega[2], 0.215300419748, atol=1.0e-07)
    assert np.allclose(deltaT[3]["A"], -0.002116364308576001, atol=1.0e-07)
    assert np.allclose(deltaT[3]["D"], -0.0027782457217954415, atol=1.0e-07)

    assert np.allclose(omega[3], 0.239230663416, atol=1.0e-07)
    assert np.allclose(deltaT[4]["A"], -0.0024842460146592134, atol=1.0e-07)
    assert np.allclose(deltaT[4]["D"], -0.003755613810180854, atol=1.0e-07)

    assert np.allclose(omega[4], 0.283506870102, atol=1.0e-07)
    assert np.allclose(deltaT[5]["A"], -0.0012994933017403724, atol=1.0e-07)
    assert np.allclose(deltaT[5]["D"], -0.002034863697140622, atol=1.0e-07)

if __name__ == "__main__":
    test_creom23_ch2()
