from pathlib import Path
import numpy as np
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_lefteomcc_calc, run_eom_correction

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eomccsdtastar_chplus():

    fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/chplus.FCIDUMP", 6, 26, 0)

    T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')

    T, H = get_hbar(T, fock, g, o, v, method='ccsdta')
    H1, H2 = H

    R0, omega0 = run_guess(H1, H2, o, v, 15, method="cisd", nacto=6, nactu=6, mult=1)

    R, omega, r0 = run_eomcc_calc(R0, omega0, T, H1, H2, o, v, method='eomccsd', state_index=[0, 6, 7], maxit=80)
    L, omega_left = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eomccsd', maxit=80)

    delta_T = []
    for i in range(len(R)):
        delta_T.append(run_eom_correction(T, R[i], L[i], r0[i], omega[i], fock, H1, H2, o, v, method="eomccsdta_star", g=g))

    #
    # Check the results
    #
    assert np.allclose(Ecorr, -0.114901981160, atol=1.0e-07)

    #
    # Compare to the results in Table III in Matthews, Stanton, JCP 145, 124102 (2016)
    #
    exc_energy_ccsdtq = [3.23, 6.96, 8.55]
    # expected = [<0.01 eV, +0.31 eV, +0.22 eV]
    for i, e in enumerate(exc_energy_ccsdtq):
        exc_energy_ev = (delta_T[i]['A'] + omega[i]) * 27.2114
        print(f"EOMCCSD(T)a* = {np.round(exc_energy_ev, 4)} eV     Error rel. CCSDTQ = {np.round(exc_energy_ev - e, 4)} eV")

if __name__ == "__main__":
    test_eomccsdtastar_chplus()


