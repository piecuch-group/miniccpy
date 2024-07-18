import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, run_lefteomcc_calc, get_hbar, run_ip_correction

def test_ipeomccsdtastar_ohminus():

        basis = '6-31g'
        nfrozen = 0

        # Define molecule geometry and basis set
        geom = [['H', (0.0, 0.0, -0.8)],
                ['O', (0.0, 0.0,  0.8)]]

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, charge=-1, unit="Angstrom", symmetry="C2V")

        T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')

        T, H = get_hbar(T, fock, g, o, v, method='ccsdta')
        H1, H2 = H

        R, omega_guess = run_guess(H1, H2, o, v, 10, method="ipcis")
        R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="ipeom2", state_index=[0, 4])
        L, omega = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method="left_ipeom2")

        delta_T = []
        for i in range(len(omega)):
            # use this version for L vectors
            delta_T.append(run_ip_correction(T, R[i], L[i], omega[i], fock, g, H1, H2, o, v, method="ipeomccsdta_star"))
            # use this version for approximating L by R*
            #delta_T.append(run_ip_correction(T, R[i], None, omega[i], fock, g, H1, H2, o, v, method="ipeomccsdta_star"))

        #
        # Check the results (obtained from PySCF)
        #
        expected_vee = []
        expected_vee_star = [-0.0033479362729279495, 0.12214287988940269]
        assert np.allclose(Ecorr, -0.1514644845729439, atol=1.0e-07)
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
#         geom = [['H', (0.0, 0.0, -0.8)],
#                 ['O', (0.0, 0.0,  0.8)]]
#
#         mol = gto.M(atom=geom, basis=basis, spin=0, symmetry="C2V", charge=-1, unit="Angstrom")
#         mf = scf.RHF(mol)
#         mf.kernel()
#
#         mycc = cc.GCCSD(mf)
#         mycc.conv_tol_normt = 1.0e-8
#         mycc.conv_tol = 1.0e-8
#         mycc.kernel()
#
#         # This is IP-EOMCCSD*
#         # myeom = eom_gccsd.EOMIP(mycc)
#         # myeom.conv_tol = 1.0e-08
#         # e = myeom.ipccsd_star(nroots=10)
#         # print(e)
#       
#         # This is IP-EOMCCSDT(a)*
#         myeom = eom_gccsd.EOMIP_Ta(mycc)
#         myeom.conv_tol = 1.0e-08
#         e = myeom.ipccsd_star(nroots=10)
#         print(e)

if __name__ == "__main__":
        test_ipeomccsdtastar_ohminus()
