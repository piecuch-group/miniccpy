from pathlib import Path
import numpy as np
from miniccpy.driver import run_scf_gamess, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eomrccsdta_chplus():

    fock, g, e_hf, o, v = run_scf_gamess(TEST_DATA_DIR + "/chplus.FCIDUMP", 6, 26, 0, rhf=True)

    T, Ecorr  = run_cc_calc(fock, g, o, v, method='rccsd')

    T, H = get_hbar(T, fock, g, o, v, method='rccsdta')
    H1, H2 = H

    R0, omega0 = run_guess(H1, H2, o, v, 10, method="rcisd", nacto=3, nactu=3)
    R, omega, r0 = run_eomcc_calc(R0, omega0, T, H1, H2, o, v, method='eomrccsdta', state_index=[0, 2, 3], maxit=200, max_size=30, fock=fock, g=g)

    expected_vee = [0.118441250230, 0.256546114945, 0.316112185013]
    for vee_calc, vee_expected in zip(omega, expected_vee):
       assert np.allclose(vee_expected, vee_calc, atol=1.0e-08)

    #
    # Compare to the results in Table III in Matthews, Stanton, JCP 145, 124102 (2016)
    #
    exc_energy_ccsdtq = [3.23, 6.96, 8.55]
    # expected = [<0.01 eV, +0.31 eV, +0.22 eV]
    for i, e in enumerate(exc_energy_ccsdtq):
        exc_energy_ev = omega[i] * 27.2114
        print(f"EOMCCSD(T)a = {np.round(exc_energy_ev, 4)} eV     Error rel. CCSDTQ = {np.round(exc_energy_ev - e, 4)} eV")

if __name__ == "__main__":
    test_eomrccsdta_chplus()





