import numpy as np
from pyscf import gto
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, run_lefteomcc_calc, get_hbar, run_ip_correction

def test_ipeomccsdtastar_ohminus():

        nfrozen = 0

        # Define molecule geometry and basis set
        geom = [['N', (0.0, 0.0, 0.0)],
                ['N', (0.0, 0.0, 1.094)]]

        ANO_basis = {'N': gto.basis.parse('''
                N    S
                  45831.05000000             0.0001578820          -0.0000363109
                   6803.94200000             0.0008911159          -0.0002053721
                   1441.54300000             0.0045553868          -0.0010499427
                    394.11230000             0.0184020951          -0.0042907952
                    128.27310000             0.0609841213          -0.0144694657
                     46.66854000             0.1633385951          -0.0412070835
                     18.09257000             0.3350192053          -0.0954241730
                      7.21535200             0.4092201761          -0.1675980653
                      2.88655900             0.1592095494          -0.1051411850
                      1.13830700             0.0011237650           0.2908298564
                      0.43702200             0.0036104375           0.6379786434
                      0.16175280            -0.0012273272           0.2327506567
                      0.05986878             0.0005036816           0.0056985008
                N    P
                     87.70229000             0.0023374202
                     20.52640000             0.0174254214
                      6.32223800             0.0756311285
                      2.25564100             0.2223630326
                      0.85611830             0.4123022954
                      0.32715340             0.3933480030
                      0.12125640             0.1103029751
                      0.04494256            -0.0028943696
        ''')}

        fock, g, e_hf, o, v = run_scf(geom, "dz", nfrozen, charge=0, unit="Angstrom", symmetry="D2H")

        T, Ecorr = run_cc_calc(fock, g, o, v, method='ccsd')

        T, H = get_hbar(T, fock, g, o, v, method='ccsdta')
        H1, H2 = H

        R, omega_guess = run_guess(H1, H2, o, v, 14, method="ipcis")
        R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="ipeom2", state_index=[0, 2, 6, 8, 10, 13], max_size=100)
        L, omega = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method="left_ipeom2", max_size=100)

        delta_T = []
        for i in range(len(R)):
            delta_T.append(run_ip_correction(T, R[i], L[i], omega[i], fock, g, H1, H2, o, v, method="ipeomccsdta_star"))

        for i in range(len(omega)):
            print(f"Root {i + 1}: IP = {omega[i] * 27.2114}")

def test_pyscf():
        from pyscf import gto, scf, cc
        from pyscf.cc import gccsd, eom_gccsd

        geom = [['N', (0.0, 0.0, 0.0)],
                ['N', (0.0, 0.0, 1.094)]]

        mol = gto.M(atom=geom, basis="dz", charge=0, unit="Angstrom", spin=0, symmetry="D2H")
        mf = scf.RHF(mol)
        mf.kernel()

        mycc = cc.GCCSD(mf).run()

        myeom = eom_gccsd.EOMIP(mycc)

        e,v = myeom.ipccsd(nroots=5)
        e,lv = myeom.ipccsd(nroots=5, left=True)
        e = myeom.ipccsd_star_contract(e, v, lv)

        print([omega * 27.2114 for omega in e])

        # Root 1: IP = 14.756635632622698
        # Root 2: IP = 17.452583879305767
        # Root 3: IP = 17.944464032881353
        # Root 4: IP = 39.956272372356885
        # Root 5: IP = 414.3445662364297
        # Root 6: IP = 414.4419718449238


if __name__ == "__main__":
        test_ipeomccsdtastar_ohminus()
        # test_pyscf()
# [14.582534562616484, 14.582534562616493, 16.75874730019113, 16.75874730019113, 16.758747300191242]



