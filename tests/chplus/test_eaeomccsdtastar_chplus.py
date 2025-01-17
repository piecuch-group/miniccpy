import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, run_lefteomcc_calc, get_hbar, run_ea_correction

def test_eaeomccsdtastar_chplus():

    basis = '6-31g'
    nfrozen = 0
    geom = [["C", (0.0, 0.0, 0.0)],
           ["H", (0.0, 0.0, 2.13713)]]

    fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, charge=1, unit="Bohr", symmetry="C2V")

    T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')
    T, H = get_hbar(T, fock, g, o, v, method='ccsdta')
    H1, H2 = H

    R, omega_guess = run_guess(H1, H2, o, v, 20, method="eacis")
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method='eaeom2', state_index=[0, 4])
    L, omega = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eaeom2')

    delta_T = []
    for i in range(len(omega)):
        # use this version for L vectors
        delta_T.append(run_ea_correction(T, R[i], L[i], omega[i], fock, g, H1, H2, o, v, method="eaeomccsdta_star"))
        # use this version for approximating L by R*
        # delta_T.append(run_ea_correction(T, R[i], None, omega[i], fock, g, H1, H2, o, v, method="eaeomccsdta_star"))

    #
    # Check the results (obtained from PySCF)
    #
    expected_vee = []
    expected_vee_star = [-0.3641487642700136, -0.06619688623488419]
    assert np.allclose(Ecorr, -0.07170350442278708, atol=1.0e-07)
    for i in range(2):
        assert np.allclose(omega[i] + delta_T[i]["A"], expected_vee_star[i], atol=1.0e-07)

# def test_pyscf():
#         from pyscf import gto, scf, cc, ao2mo
#         from pyscf.cc import gccsd, eom_gccsd, gintermediates
#
#         basis = '6-31g'
#         nfrozen = 0
#
#         # Define molecule geometry and basis set
#         geom = [["C", (0.0, 0.0, 0.0)],
#                 ["H", (0.0, 0.0, 2.13713)]]
#
#         mol = gto.M(atom=geom, basis=basis, spin=0, symmetry="C2V", charge=1, unit="Bohr")
#         mf = scf.RHF(mol)
#         mf.kernel()
#
#         mycc = cc.GCCSD(mf)
#         mycc.conv_tol_normt = 1.0e-8
#         mycc.conv_tol = 1.0e-8
#         mycc.kernel()
#
#         # This is EA-EOMCCSDT(a)*
#         myeom = eom_gccsd.EOMEA_Ta(mycc)
#         myeom.conv_tol = 1.0e-08
#         e = myeom.eaccsd_star(nroots=20)
#         print(e)

if __name__ == "__main__":
    test_eaeomccsdtastar_chplus()

