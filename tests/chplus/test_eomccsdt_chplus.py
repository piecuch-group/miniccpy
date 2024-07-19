from pathlib import Path
import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_lefteomcc_calc, run_eom_correction

def test_eomccsdt_chplus():

    basis = '6-31g'
    nfrozen = 0
    geom = [["C", (0.0, 0.0, 0.0)],
            ["H", (0.0, 0.0, 2.13713)]]

    fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, charge=1, unit="Bohr", symmetry="C2V")

    T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsdt')

    H1, H2 = get_hbar(T, fock, g, o, v, method='ccsdt')

    #R0, omega0 = run_guess(H1, H2, o, v, 15, method="cisd", nacto=6, nactu=6, mult=1)
    R0, omega0 = run_guess(H1, H2, o, v, 15, method="cis", mult=1)

    R, omega, r0 = run_eomcc_calc(R0, omega0, T, H1, H2, o, v, method='eomccsdt', state_index=[0, 6, 7], maxit=80)

if __name__ == "__main__":
    test_eomccsdt_chplus()


