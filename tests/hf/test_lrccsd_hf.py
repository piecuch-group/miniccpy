import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, get_hbar, run_lrcc1_calc
from miniccpy.integrals import spatial_to_spinorb_onebody

from pyscf import gto, scf

def test_lrccsd_h2o():

    basis = 'dzp'
    nfrozen = 1

    # Define molecule geometry and basis set
    geom = [['H', (0.0, 0.0, -1.7330)],
            ['F', (0.0, 0.0,  1.7330)]]

    fock, g, e_hf, o, v, mu = run_scf(geom, basis, nfrozen, unit="Bohr", symmetry="C2V", cartesian=True, multipole=2)
    T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')
    H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

    mu_ref = np.einsum("xyii->xy", mu[:, :, o, o])
    eta = [[None for _ in range(3)] for _ in range(3)]
    mu_corr = np.zeros((3, 3))
    cnt = 0
    for i in range(3):
        for j in range(3):
            eta[i][j], mu_corr[i, j] = run_lrcc1_calc(T, H1, H2, mu[i, j, :, :], o, v, method="lrccsd")

    print("Reference Quadrupole Moment")
    print(f"   xx: {mu_ref[0, 0]}")
    print(f"   yy: {mu_ref[1, 1]}")
    print(f"   zz: {mu_ref[2, 2]}")
    print(f"   xz: {mu_ref[0, 2]}     zx: {mu_ref[2, 0]}")
    print(f"   xy: {mu_ref[0, 1]}     yx: {mu_ref[1, 0]}")
    print(f"   yz: {mu_ref[1, 2]}     zy: {mu_ref[1, 2]}")
    print("LR-CCSD Quadrupole Moment")
    mu_ccsd = mu_ref + mu_corr
    print(f"   xx: {mu_ccsd[0, 0]}")
    print(f"   yy: {mu_ccsd[1, 1]}")
    print(f"   zz: {mu_ccsd[2, 2]}")
    print(f"   xz: {mu_ccsd[0, 2]}     zx: {mu_ccsd[2, 0]}")
    print(f"   xy: {mu_ccsd[0, 1]}     yx: {mu_ccsd[1, 0]}")
    print(f"   yz: {mu_ccsd[1, 2]}     zy: {mu_ccsd[1, 2]}")

if __name__ == "__main__":
    test_lrccsd_h2o()


