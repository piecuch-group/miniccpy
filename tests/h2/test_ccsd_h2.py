#import pyscf
import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_leftcc_calc, get_hbar, run_cmx_calc

def test_cmx2_h2():

        basis = 'cc-pvdz'
        nfrozen = 0
        Re = 0.74 * 2

        geom = [['H', (0., 0., -Re)],
                ['H', (0., 0., Re)]]

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200)

        T, E_corr = run_cc_calc(fock, g, o, v, method='ccsd', maxit=80)
        H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')
        L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_ccsd")
        cmx_corr = run_cmx_calc(T, L, E_corr, H1, H2, o, v, method="cmx2_ccsd")

        #
        # Check the results
        #
        assert np.allclose(E_corr, -0.063257773872, atol=1.0e-07)

        # Just make sure that CCSD = FCI for a 2-electron problem
        #mol = pyscf.M(atom=geom, basis=basis, symmetry=True, spin=0, unit="Bohr")
        #mf = mol.RHF().run()
        #
        # create an FCI solver based on the SCF object
        #
        #cisolver = pyscf.fci.FCI(mf)
        #print('E(FCI) = %.12f' % cisolver.kernel()[0])
        #print(f"E(CCSD) = {E_corr + e_hf}")

if __name__ == "__main__":
        test_cmx2_h2()




