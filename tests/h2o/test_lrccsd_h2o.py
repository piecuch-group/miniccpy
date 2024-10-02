import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, get_hbar, run_lrcc1_calc
from miniccpy.integrals import spatial_to_spinorb_onebody

from pyscf import gto, scf

def test_lrccsd_h2o():

    basis = '6-31g'
    nfrozen = 0

    re = 1

    # Define molecule geometry and basis set
    if re == 1:
        geom = [['H', (0, 1.515263, -1.058898)],
                ['H', (0, -1.515263, -1.058898)],
                ['O', (0.0, 0.0, -0.0090)]]
    elif re == 2:
        geom = [["O", (0.0, 0.0, -0.0180)],
                ["H", (0.0, 3.030526, -2.117796)],
                ["H", (0.0, -3.030526, -2.117796)]]

    fock, g, e_hf, o, v, mu = run_scf(geom, basis, nfrozen, multipole=1)
    T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')
    H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

    mu_ref = np.einsum("xii->x", mu[:, o, o])
    eta = [None for _ in range(3)]
    mu_corr = np.zeros(3)
    for i in range(3):
        eta[i], mu_corr[i] = run_lrcc1_calc(T, H1, H2, mu[i, :, :], o, v, method="lrccsd")

    print("Reference Dipole Moment")
    print(f"   x: {mu_ref[0]}")
    print(f"   y: {mu_ref[1]}")
    print(f"   z: {mu_ref[2]}")
    print("LR-CCSD Dipole Moment")
    mu_ccsd = mu_ref + mu_corr
    print(f"   x: {mu_ccsd[0]}")
    print(f"   y: {mu_ccsd[1]}")
    print(f"   z: {mu_ccsd[2]}")

if __name__ == "__main__":
    test_lrccsd_h2o()


