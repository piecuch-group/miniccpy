import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_scf, run_cc_calc, get_hbar, run_leftcc_calc
from miniccpy.rdm1 import rdm1_ccsd
from miniccpy.rdm2 import rdm2_ccsd, rdm2_ccsd_fact
from miniccpy.energy import cc_corr_energy_from_rdm


def test_hf_rdmtoe():
        
        basis = 'sto-6g'
        nfrozen = 0

        geom = [['H', (0.0, 0.0, -0.8)],
                ['F', (0.0, 0.0,  0.8)]]
        

        fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200, unit="Angstrom")
        
        T, E_corr = run_cc_calc(fock, g, o, v, method='ccsd', maxit=80)

        H1, H2 = get_hbar(T, fock, g, o, v, method="ccsd")

        L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_ccsd")

        rdm1 = rdm1_ccsd(L, T)
        rdm2 = rdm2_ccsd(L, T)
        rdm2_fact = rdm2_ccsd_fact(L, T)

        E_corr_from_rdm = cc_corr_energy_from_rdm(rdm1, rdm2, fock, g, o, v)
        E_corr_from_rdm_fact = cc_corr_energy_from_rdm(rdm1, rdm2_fact, fock, g, o, v)
        
        assert np.allclose(E_corr_from_rdm, E_corr, atol=1.0e-07)
        assert np.allclose(E_corr_from_rdm_fact, E_corr, atol=1.0e-07)

if __name__ == "__main__":
    test_hf_rdmtoe()



