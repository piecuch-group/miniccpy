"""CCSD(T) computation for the stretched F2 molecule at at
interatomic distance of R = 2Re, where Re = 2.66816 bohr,
described using the cc-pVTZ basis set.
Reference: Chem. Phys. Lett. 344, 165 (2001)."""

import numpy as np
from pyscf import scf, gto
from ccpy.drivers.driver import Driver
from copy import deepcopy

def test_crcc23_f2():
    geometry = [["F", (0.0, 0.0, -2.66816)],
                ["F", (0.0, 0.0, 2.66816)]]
    mol = gto.M(
        atom=geometry,
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=2)
    driver.run_cc(method="ccsd")
   
    T = driver.T
    H0 = deepcopy(driver.hamiltonian)

    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    L = driver.L[0]

    #hbar_vvov = H0.ab.vvov + (
    #            -np.einsum("mbie,am->abie", H0.ab.ovov, T.a, optimize=True) # (2)
    #            +np.einsum("nbfe,afin->abie", H0.ab.ovvv, T.aa, optimize=True) # (3)
    #            -np.einsum("nmfe,bm,afin->abie", H0.ab.oovv, T.b, T.aa, optimize=True) # (4)
    #            +np.einsum("bnef,afin->abie", H0.bb.vovv, T.ab, optimize=True) # (5)
    #            -np.einsum("mnef,bm,afin->abie", H0.bb.oovv, T.b, T.ab, optimize=True) # (6)
    #            -np.einsum("anfe,fbin->abie", H0.ab.vovv, T.ab, optimize=True) # (7)
    #            +np.einsum("mnfe,am,fbin->abie", H0.ab.oovv, T.a, T.ab, optimize=True) # (8)
    #            -np.einsum("amie,bm->abie", H0.ab.voov, T.b, optimize=True) # (9)
    #            +np.einsum("nmie,abnm->abie", H0.ab.ooov, T.ab, optimize=True) # (10)
    #            +np.einsum("mnie,am,bn->abie", H0.ab.ooov, T.a, T.b, optimize=True) # (11)
    #            -np.einsum("me,abim->abie", driver.hamiltonian.b.ov, T.ab, optimize=True) # (12)
    #            # (vvvv) parts
    #)

    ##############################
    ### Overall RHF Expression ###
    ##############################
    LH.b += 2.0 * np.einsum("fena,feni->ai", H.ab.vvov, L.ab, optimize=True)
    LH.b -= np.einsum("efna,efin->ai", H.ab.vvov, L.ab, optimize=True)

    ################
    ### Standard ###
    ################
    # Quasi-Hbar intermediates
    I2B_voov = H0.ab.voov + (
                +np.einsum("nmfe,afin->amie", H0.ab.oovv, T.aa, optimize=True)
                +np.einsum("nmfe,afin->amie", H0.bb.oovv, T.ab, optimize=True)
                -np.einsum("nmie,an->amie", H0.ab.ooov, T.a, optimize=True)
    )
    I2B_ovov = H0.ab.ovov - np.einsum("mnfe,fbin->mbie", H0.ab.oovv, T.ab, optimize=True)
    hbar2B_vvov = H0.ab.vvov + (
                -np.einsum("mbie,am->abie", I2B_ovov, T.a, optimize=True) # (2) + (8)
                -np.einsum("amie,bm->abie", I2B_voov, T.b, optimize=True) # (4) + (6) + (9) + (11)
                +np.einsum("nbfe,afin->abie", H0.ab.ovvv, T.aa, optimize=True) # (3)
                +np.einsum("bnef,afin->abie", H0.bb.vovv, T.ab, optimize=True) # (5)
                -np.einsum("anfe,fbin->abie", H0.ab.vovv, T.ab, optimize=True) # (7)
                +np.einsum("nmie,abnm->abie", H0.ab.ooov, T.ab, optimize=True) # (10)
                -np.einsum("me,abim->abie", driver.hamiltonian.b.ov, T.ab, optimize=True) # (12)
               # (vvvv) parts
    )
    LH_exact = np.einsum("fena,feni->ai", hbar2B_vvov, L.ab, optimize=True)
    #LH_exact += 0.5 * np.einsum("fena,efin->ai", H.bb.vvov, L.bb, optimize=True)

    ####################
    ### Refactorized ###
    ####################
    # Quasi-Hbar intermediates
    I2B_voov = H0.ab.voov + (
                +np.einsum("nmfe,afin->amie", H0.ab.oovv, T.aa, optimize=True)
                +np.einsum("nmfe,afin->amie", H0.bb.oovv, T.ab, optimize=True)
                -np.einsum("nmie,an->amie", H0.ab.ooov, T.a, optimize=True)
    )
    I2B_ovov = H0.ab.ovov - np.einsum("mnfe,fbin->mbie", H0.ab.oovv, T.ab, optimize=True)
    # L*T intermediates
    x2B_oovo = np.einsum("feni,em->nifm", L.ab, T.b, optimize=True) # CPU: V^2O^3
    x2B_ooov = np.einsum("efni,em->nimf", L.ab, T.a, optimize=True) # CPU: V^2O^3
    x2B_voov = np.einsum("egmj,efmn->fjng", L.ab, T.aa, optimize=True)
    x2C_voov = np.einsum("egmj,efmn->fjng", L.ab, T.ab, optimize=True)
    x2B_vovo = -np.einsum("geni,fenm->figm", L.ab, T.ab, optimize=True)
    x2B_oooo = np.einsum("feoi,fenm->oinm", L.ab, T.ab, optimize=True)
    x1B_oo = np.einsum("feni,fenm->im", L.ab, T.ab, optimize=True)
    # (I) - direct term
    LH_refactor_I = np.einsum("fena,feni->ai", H0.ab.vvov, L.ab, optimize=True)             # (1) CPU: V^3O^2
    LH_refactor_I -= np.einsum("nifm,fmna->ai", x2B_oovo, I2B_voov, optimize=True)          # (4) + (6) + (9) + (11) CPU: V^2O^3
    LH_refactor_I -= np.einsum("nimf,mfna->ai", x2B_ooov, I2B_ovov, optimize=True)          # (2) + (8) CPU: V^2O^3
    LH_refactor_I += np.einsum("fing,ngfa->ai", x2B_voov, H0.ab.ovvv, optimize=True)        # (3) CPU: V^3O^2 
    LH_refactor_I += (
                        np.einsum("fing,gnaf->ai", x2C_voov, H0.ab.vovv, optimize=True)          # (5) CPU: V^3O^2
                        - np.einsum("fing,gnfa->ai", x2C_voov, H0.ab.vovv, optimize=True)        # (5) CPU: V^3O^2
    )
    LH_refactor_I += np.einsum("figm,gmfa->ai", x2B_vovo, H0.ab.vovv, optimize=True)        # (7) CPU: V^3O^2
    LH_refactor_I += np.einsum("oinm,nmoa->ai", x2B_oooo, H0.ab.ooov, optimize=True)        # (10) CPU: VO^4
    LH_refactor_I -= np.einsum("im,ma->ai", x1B_oo, driver.hamiltonian.b.ov, optimize=True) # (12) CPU: VO^2
    # (II) - exchange term
    LH_refactor_II = np.einsum("efna,efin->ai", H0.ab.vvov, L.ab, optimize=True)
    #LH_refactor_II -= np.einsum("nimf,fmna->ai", x2B_ooov, I2B_voov, optimize=True)  # CPU: V^2O^3
    #LH_refactor_II -= np.einsum("nifm,mfna->ai", x2B_oovo, I2B_ovov, optimize=True)  # CPU: V^2O^3

    LH_refactor = LH_refactor_I + 0.0 * LH_refactor_II

    # Check error
    print("Error = ", np.linalg.norm(LH_exact.flatten() - LH_refactor.flatten()))
    




if __name__ == "__main__":
    test_crcc23_f2()
