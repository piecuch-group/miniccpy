import numpy as np
from miniccpy.pspace import get_active_4h2p_pspace
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_dip_correction, run_lefteomcc_calc

def test_dipeom4_cl2():

    basis = '6-31g'
    nfrozen = 10

    geom = [["Cl", (0.0, 0.0, 0.0)],
            ["Cl", (0.0, 0.0, 1.9870)]]

    fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, symmetry="D2H", unit="Angstrom", cartesian=False, charge=0, x2c=True)

    T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')
    T, H = get_hbar(T, fock, g, o, v, method='ccsdta')
    H1, H2 = H
    #H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

    nroot = 10
    R, omega_guess = run_guess(H1, H2, o, v, nroot, method="dipcis")
    # Set up the list of 4h2p excitations corresponding to the active-space DIP-EOMCCSD(4h-2p){No} method
    no, nu = fock[o, v].shape
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="dipeom3", state_index=[3])
    L, omega = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method="left_dipeom3")
    for i in range(len(R)):
        #delta_star = run_dip_correction(T, R[i], None, omega[i], fock, g, H1, H2, o, v, method="dipeom4_star")
        delta_star = run_dip_correction(T, R[i], L[i], omega[i], fock, g, H1, H2, o, v, method="dipeom34")

if __name__ == "__main__":
    test_dipeom4_cl2()
