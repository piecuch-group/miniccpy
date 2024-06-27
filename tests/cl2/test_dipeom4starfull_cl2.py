import numpy as np
from miniccpy.pspace import get_active_4h2p_pspace
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_dip_correction

def test_dipeom4_cl2():

    basis = '6-31g'
    nfrozen = 10

    geom = [["Cl", (0.0, 0.0, 0.0)],
            ["Cl", (0.0, 0.0, 1.9870)]]

    fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, symmetry="D2H", unit="Angstrom", cartesian=False, charge=0, x2c=True)

    T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')
    H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

    nroot = 10
    R, omega_guess = run_guess(H1, H2, o, v, nroot, method="dipcis")
    # Set up the list of 4h2p excitations corresponding to the active-space DIP-EOMCCSD(4h-2p){No} method
    no, nu = fock[o, v].shape
    # Here, we are using 14 active occupied spinorbitals, which corresponds to full DIP-EOMCCSD(4h-2p)
    r3_excitations = get_active_4h2p_pspace(no, nu, nacto=14)
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="dipeom4_star_p", state_index=[0, 1, 2, 3, 4, 5, 6], out_of_core=True, r3_excitations=r3_excitations, fock=fock, g=g)

if __name__ == "__main__":
    test_dipeom4_cl2()
