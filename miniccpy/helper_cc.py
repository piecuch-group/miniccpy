import numpy as np

def get_rccsd_intermediates(t1, t2, f, g, o, v):
    norbitals = f.shape[0]
    nunocc, nocc = f[v, o].shape
    n1 = nunocc * nocc

    t2s = t2 - np.transpose(t2, (0, 1, 3, 2))
    gs_oovv = g[o, o, v, v] - np.transpose(g[o, o, v, v], (0, 1, 3, 2))
    gs_ooov = g[o, o, o, v] - np.transpose(g[o, o, o, v], (1, 0, 2, 3))

    H1 = np.zeros((norbitals, norbitals))
    H2 = np.zeros((norbitals, norbitals, norbitals, norbitals))

    H1[o, v] = f[o, v] + (
                2.0 * np.einsum("imae,em->ia", g[o, o, v, v], t1, optimize=True)
                - np.einsum("imea,em->ia", g[o, o, v, v], t1, optimize=True)
    )

    H1[o, o] = f[o, o] + (
                np.einsum("je,ei->ji", H1[o, v], t1, optimize=True)
                + np.einsum("jmie,em->ji", gs_ooov, t1, optimize=True)
                + np.einsum("jmie,em->ji", g[o, o, o, v], t1, optimize=True)
                + 0.5 * np.einsum("jnef,efin->ji", gs_oovv, t2s, optimize=True)
                + np.einsum("jnef,efin->ji", g[o, o, v, v], t2, optimize=True)
    )

    H1[v, v] = f[v, v] + (
                - np.einsum("mb,am->ab", H1[o, v], t1, optimize=True)
                + np.einsum("ambe,em->ab", g[v, o, v, v], t1, optimize=True)
                - np.einsum("ameb,em->ab", g[v, o, v, v], t1, optimize=True)
                + np.einsum("ambe,em->ab", g[v, o, v, v], t1, optimize=True)
                - 0.5 * np.einsum("mnbf,afmn->ab", gs_oovv, t2s, optimize=True)
                - np.einsum("mnbf,afmn->ab", g[o, o, v, v], t2, optimize=True)
    )

    Q1 = -np.einsum("nmef,an->amef", g[o, o, v, v], t1, optimize=True)
    I_vovv = g[v, o, v, v] + 0.5 * Q1
    H2[v, o, v, v] = I_vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    I_ooov = g[o, o, o, v] + 0.5 * Q1
    H2[o, o, o, v] = I_ooov + 0.5 * Q1

    Q1 = -np.einsum("mnef,an->maef", g[o, o, v, v], t1, optimize=True)
    I_ovvv = g[o, v, v, v] + 0.5 * Q1
    H2[o, v, v, v] = I_ovvv + 0.5 * Q1

    Q1 = np.einsum("nmef,fi->nmei", g[o, o, v, v], t1, optimize=True)
    I_oovo = g[o, o, v, o] + 0.5 * Q1
    H2[o, o, v, o] = I_oovo + 0.5 * Q1


    H2[v, v, v, v] = g[v, v, v, v] + (
                - np.einsum("mbef,am->abef", I_ovvv, t1, optimize=True)
                - np.einsum("amef,bm->abef", I_vovv, t1, optimize=True)
                + np.einsum("mnef,abmn->abef", g[o, o, v, v], t2, optimize=True)
    )

    H2[o, o, o, o] = g[o, o, o, o] + (
                np.einsum("mnej,ei->mnij", I_oovo, t1, optimize=True)
                + np.einsum("mnie,ej->mnij", I_ooov, t1, optimize=True)
                + np.einsum("mnef,efij->mnij", g[o, o, v, v], t2, optimize=True)
    )

    H2[o, o, v, v] = g[o, o, v, v].copy()

    H2[v, o, o, v] = g[v, o, o, v] + (
                np.einsum("amfe,fi->amie", I_vovv, t1, optimize=True)
                - np.einsum("nmie,an->amie", I_ooov, t1, optimize=True)
                + np.einsum("nmfe,afin->amie", g[o, o, v, v], t2s, optimize=True)
                + np.einsum("nmfe,afin->amie", gs_oovv, t2, optimize=True)
    )

    H2[o, v, v, o] = g[o, v, v, o] + (
                np.einsum("maef,fi->maei", I_ovvv, t1, optimize=True)
                - np.einsum("mnei,an->maei", I_oovo, t1, optimize=True)
                + np.einsum("mnef,afin->maei", g[o, o, v, v], t2s, optimize=True)
                + np.einsum("mnef,fani->maei", gs_oovv, t2, optimize=True)
    )

    H2[o, v, o, v] = g[o, v, o, v] + (
                np.einsum("mafe,fi->maie", I_ovvv, t1, optimize=True)
                - np.einsum("mnie,an->maie", I_ooov, t1, optimize=True)
                - np.einsum("mnfe,fain->maie", g[o, o, v, v], t2, optimize=True)
    )

    H2[v, o, v, o] = g[v, o, v, o] + (
                - np.einsum("nmei,an->amei", I_oovo, t1, optimize=True)
                + np.einsum("amef,fi->amei", I_vovv, t1, optimize=True)
                - np.einsum("nmef,afni->amei", g[o, o, v, v], t2, optimize=True)
    )

    Is_ooov = H2[o, o, o, v] - np.transpose(H2[o, o, o, v], (1, 0, 2, 3))

    Q1 = g[v, o, o, v] + np.einsum("amfe,fi->amie", g[v, o, v, v], t1, optimize=True)
    H2[v, o, o, o] = g[v, o, o, o] + (
                np.einsum("me,aeij->amij", H1[o, v], t2, optimize=True)
                - np.einsum("nmij,an->amij", H2[o, o, o, o], t1, optimize=True)
                + np.einsum("mnjf,afin->amij", Is_ooov, t2, optimize=True)
                + np.einsum("nmfj,afin->amij", H2[o, o, v, o], t2s, optimize=True)
                - np.einsum("nmif,afnj->amij", H2[o, o, o, v], t2, optimize=True)
                + np.einsum("amej,ei->amij", g[v, o, v, o], t1, optimize=True)
                + np.einsum("amie,ej->amij", Q1, t1, optimize=True)
                + np.einsum("amef,efij->amij", g[v, o, v, v], t2, optimize=True)
    )

    Q1 = g[o, v, o, v] + np.einsum("mafe,fj->maje", g[o, v, v, v], t1, optimize=True)
    H2[o, v, o, o] = g[o, v, o, o] + (
                np.einsum("me,eaji->maji", H1[o, v], t2, optimize=True)
                - np.einsum("mnji,an->maji", H2[o, o, o, o], t1, optimize=True)
                + np.einsum("mnjf,fani->maji", Is_ooov, t2, optimize=True)
                + np.einsum("mnjf,fani->maji", H2[o, o, o, v], t2s, optimize=True)
                - np.einsum("mnfi,fajn->maji", H2[o, o, v, o], t2, optimize=True)
                + np.einsum("maje,ei->maji", Q1, t1, optimize=True)
                + np.einsum("maei,ej->maji", g[o, v, v, o], t1, optimize=True)
                + np.einsum("mafe,feji->maji", g[o, v, v, v], t2, optimize=True)
    )

    Q1 = g[o, v, o, v] - np.einsum("mnie,bn->mbie", g[o, o, o, v], t1, optimize=True)
    Q1 = -np.einsum("mbie,am->abie", Q1, t1, optimize=True)
    H2[v, v, o, v] = g[v, v, o, v] + Q1 + (
                - np.einsum("me,abim->abie", H1[o, v], t2, optimize=True)
                + np.einsum("abfe,fi->abie", H2[v, v, v, v], t1, optimize=True)
                + np.einsum("nbfe,afin->abie", H2[o, v, v, v], t2s, optimize=True)
                + np.einsum("bnef,afin->abie", H2[v, o, v, v], t2, optimize=True)
                - np.einsum("bnfe,afin->abie", H2[v, o, v, v], t2, optimize=True)
                - np.einsum("amfe,fbim->abie", H2[v, o, v, v], t2, optimize=True)
                - np.einsum("amie,bm->abie", g[v, o, o, v], t1, optimize=True)
                + np.einsum("nmie,abnm->abie", g[o, o, o, v], t2, optimize=True)
    )

    Q1 = g[v, o, v, o] - np.einsum("nmei,bn->bmei", g[o, o, v, o], t1, optimize=True)
    Q1 = -np.einsum("bmei,am->baei", Q1, t1, optimize=True)
    H2[v, v, v, o] = g[v, v, v, o] + Q1 + (
                - np.einsum("me,bami->baei", H1[o, v], t2, optimize=True)
                + np.einsum("baef,fi->baei", H2[v, v, v, v], t1, optimize=True)
                + np.einsum("bnef,fani->baei", H2[v, o, v, v], t2, optimize=True)
                - np.einsum("bnfe,fani->baei", H2[v, o, v, v], t2, optimize=True)
                + np.einsum("bnef,fani->baei", H2[v, o, v, v], t2s, optimize=True)
                - np.einsum("maef,bfmi->baei", H2[o, v, v, v], t2, optimize=True)
                - np.einsum("naei,bn->baei", g[o, v, v, o], t1, optimize=True)
                + np.einsum("nmei,banm->baei", g[o, o, v, o], t2, optimize=True)
    )
    return H1, H2


def get_rccs_intermediates(t1, f, g, o, v):
    """
    Calculate the CCS-like similarity-transformed HBar intermediates (H_N e^T1)_C.
    """
    norbitals = f.shape[0]
    # allocate arrays
    H1 = np.zeros((norbitals, norbitals))
    H2 = np.zeros((norbitals, norbitals, norbitals, norbitals))
    # 1-body intermediates
    H1[o, v] = f[o, v] + (
          np.einsum("mnef,fn->me", g[o, o, v, v], t1, optimize=True)
        - np.einsum("mnfe,fn->me", g[o, o, v, v], t1, optimize=True)
        + np.einsum("mnef,fn->me", g[o, o, v, v], t1, optimize=True)
    )
    H1[v, v] = f[v, v] + (
          np.einsum("anef,fn->ae", g[v, o, v, v], t1, optimize=True)
        - np.einsum("anfe,fn->ae", g[v, o, v, v], t1, optimize=True)
        + np.einsum("anef,fn->ae", g[v, o, v, v], t1, optimize=True)
        - np.einsum("me,am->ae", H1[o, v], t1, optimize=True)
    )
    H1[o, o] = f[o, o] + (
          np.einsum("mnif,fn->mi", g[o, o, o, v], t1, optimize=True)
        - np.einsum("nmif,fn->mi", g[o, o, o, v], t1, optimize=True)
        + np.einsum("mnif,fn->mi", g[o, o, o, v], t1, optimize=True)
        + np.einsum("me,ei->mi", H1[o, v], t1, optimize=True)
    )
    # 2-body intermediates
    H2[o, o, o, v] = np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    H2[o, o, o, o] = g[o, o, o, o] + (
          np.einsum("nmje,ei->mnij", g[o, o, o, v] + 0.5 * H2[o, o, o, v], t1, optimize=True) 
        + np.einsum("mnie,ej->mnij", g[o, o, o, v] + 0.5 * H2[o, o, o, v], t1, optimize=True)
    )
    H2[v, o, v, v] = -np.einsum("nmef,an->amef", g[o, o, v, v], t1, optimize=True)

    H2[v, o, o, v] = g[v, o, o, v] + (
          np.einsum("amfe,fi->amie", g[v, o, v, v] + 0.5 * H2[v, o, v, v], t1, optimize=True)
        - np.einsum("nmie,an->amie", g[o, o, o, v] + 0.5 * H2[o, o, o, v], t1, optimize=True)
    )
    H2[v, o, v, o] = g[v, o, v, o] - (
          np.einsum("mnie,an->amei", g[o, o, o, v] + 0.5 * H2[o, o, o, v], t1, optimize=True) 
        - np.einsum("amef,fi->amei", g[v, o, v, v] + 0.5 * H2[v, o, v, v], t1, optimize=True)
    )

    I_oooo = g[o, o, o, o] + (
          np.einsum("mnif,fj->mnij", g[o, o, o, v], t1, optimize=True)
        + np.einsum("mnej,ei->mnij", g[o, o, v, o], t1, optimize=True)
    )
    I_ovvo = g[o, v, v, o] + np.einsum("bmfe,fj->mbej", g[v, o, v, v], t1, optimize=True)
    I_voov = np.einsum("amef,ei->amif", g[v, o, v, v] + H2[v, o, v, v], t1, optimize=True)

    H2[o, v, o, o] = g[o, v, o, o] + (
          np.einsum("mbej,ei->mbij", I_ovvo, t1, optimize=True)
        - np.einsum("mnij,bn->mbij", I_oooo, t1, optimize=True)
    )
    H2[v, o, o, o] = g[v, o, o, o] + np.einsum("amif,fj->amij", g[v, o, o, v] + I_voov, t1, optimize=True)
    H2[v, v, o, v] = g[v, v, o, v] - np.einsum("bmei,am->abie", g[v, o, v, o], t1, optimize=True)
    return H1, H2

def get_ccs_intermediates(t1, f, g, o, v):
    """Calculate the quantities related to the one-
    and two-body components of the CCS similarity-transformed 
    Hamiltonian, [H_N exp(T1)]_C, which serve as suitable 
    intermediates for constructing the CCSD amplitude equations.
        H1[:, :] ~ < p | [H_N exp(T1)]_C | q > (related to, not equal!)
        H2[:, :, :, :] ~ < pq | [H_N exp(T1)]_C | rs > (related to, not equal!)
    """

    norbitals = f.shape[0]

    H1 = np.zeros((norbitals, norbitals))
    H2 = np.zeros((norbitals, norbitals, norbitals, norbitals))

    # 1-body components
    H1[o, v] = f[o, v] + np.einsum("mnef,fn->me", g[o, o, v, v], t1, optimize=True)
    H1[v, v] = f[v, v] + (
        np.einsum("anef,fn->ae", g[v, o, v, v], t1, optimize=True)
        - np.einsum("me,am->ae", H1[o, v], t1, optimize=True)
    ) 
    H1[o, o] = f[o, o] + (
        np.einsum("mnif,fn->mi", g[o, o, o, v], t1, optimize=True)
        + np.einsum("me,ei->mi", H1[o, v], t1, optimize=True)
    ) 
    # 2-body components
    H2[o, o, o, v] = np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True) 

    H2[o, o, o, o] = 0.5 * g[o, o, o, o] + np.einsum("nmje,ei->mnij", g[o, o, o, v] + 0.5 * H2[o, o, o, v], t1, optimize=True) # no(4)nu(1)
    H2[o, o, o, o] -= np.transpose(H2[o, o, o, o], (0, 1, 3, 2))

    H2[v, o, v, v] = -np.einsum("mnfe,an->amef", g[o, o, v, v], t1, optimize=True) # no(2)nu(3)

    H2[v, o, o, v] = g[v, o, o, v] + (
              np.einsum("amfe,fi->amie", g[v, o, v, v] + 0.5 * H2[v, o, v, v], t1, optimize=True)
            - np.einsum("nmie,an->amie", g[o, o, o, v] + 0.5 * H2[o, o, o, v], t1, optimize=True)
    )

    L_amie = g[v, o, o, v] + 0.5 * np.einsum('amef,ei->amif', g[v, o, v, v], t1, optimize=True) # no(2)nu(3)
    X_mnij = g[o, o, o, o] + np.einsum('mnie,ej->mnij', H2[o, o, o, v], t1, optimize=True) # no(4)nu(1)
    H2[v, o, o, o] = 0.5 * g[v, o, o, o] + (
        np.einsum('amie,ej->amij', L_amie, t1, optimize=True)
       -0.25 * np.einsum('mnij,am->anij', X_mnij, t1, optimize=True)
    ) 
    H2[v, o, o, o] -= np.transpose(H2[v, o, o, o], (0, 1, 3, 2))

    L_amie = np.einsum('mnie,am->anie', g[o, o, o, v], t1, optimize=True)
    H2[v, v, o, v] = g[v, v, o, v] + np.einsum("anie,bn->abie", L_amie, t1, optimize=True) # no(1)nu(4)

    return H1, H2


def get_ccsd_intermediates(t1, t2, f, g, o, v):
    """Calculate the quantities related to the one-
    and two-body components of the CCSD similarity-transformed 
    Hamiltonian, [H_N exp(T1+T2)]_C, which serve as suitable 
    intermediates for constructing the CCSDT amplitude equations.
        H1[:, :] ~ < p | [H_N exp(T1+T2)]_C | q > (related to, not equal!)
        H2[:, :, :, :] ~ < pq | [H_N exp(T1+T2)]_C | rs > (related to, not equal!)
    """

    norbitals = f.shape[0]

    H1 = np.zeros((norbitals, norbitals))
    H2 = np.zeros((norbitals, norbitals, norbitals, norbitals))

    # 1-body components
    H1[o, v] = f[o, v] + np.einsum("imae,em->ia", g[o, o, v, v], t1, optimize=True)

    H1[o, o] = f[o, o] + (
            np.einsum("je,ei->ji", H1[o, v], t1, optimize=True)
            + np.einsum("jmie,em->ji", g[o, o, o, v], t1, optimize=True)
            + 0.5 * np.einsum("jnef,efin->ji", g[o, o, v, v], t2, optimize=True)
    )

    H1[v, v] = f[v, v] + (
            - np.einsum("mb,am->ab", H1[o, v], t1, optimize=True)
            + np.einsum("ambe,em->ab", g[v, o, v, v], t1, optimize=True)
            - 0.5 * np.einsum("mnbf,afmn->ab", g[o, o, v, v], t2, optimize=True)
    )

    # 2-body components
    Q1 = -np.einsum("mnfe,an->amef", g[o, o, v, v], t1, optimize=True)
    I_vovv = g[v, o, v, v] + 0.5 * Q1
    H2[v, o, v, v] = I_vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    I_ooov = g[o, o, o, v] + 0.5 * Q1
    H2[o, o, o, v] = I_ooov + 0.5 * Q1

    Q1 = -np.einsum("bmfe,am->abef", I_vovv, t1, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H2[v, v, v, v] = g[v, v, v, v] + 0.5 * np.einsum("mnef,abmn->abef", g[o, o, v, v], t2, optimize=True) + Q1

    Q1 = +np.einsum("nmje,ei->mnij", I_ooov, t1, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H2[o, o, o, o] = g[o, o, o, o] + 0.5 * np.einsum("mnef,efij->mnij", g[o, o, v, v], t2, optimize=True) + Q1

    H2[v, o, o, v] = g[v, o, o, v] + (
            np.einsum("amfe,fi->amie", I_vovv, t1, optimize=True)
            - np.einsum("nmie,an->amie", I_ooov, t1, optimize=True)
            + np.einsum("nmfe,afin->amie", g[o, o, v, v], t2, optimize=True)
    )

    Q1 = np.einsum("mnjf,afin->amij", H2[o, o, o, v], t2, optimize=True)
    Q2 = g[v, o, o, v] + 0.5 * np.einsum("amef,ei->amif", g[v, o, v, v], t1, optimize=True)
    Q2 = np.einsum("amif,fj->amij", Q2, t1, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H2[v, o, o, o] = g[v, o, o, o] + Q1 + (
            np.einsum("me,aeij->amij", H1[o, v], t2, optimize=True)
            - np.einsum("nmij,an->amij", H2[o, o, o, o], t1, optimize=True)
            + 0.5 * np.einsum("amef,efij->amij", g[v, o, v, v], t2, optimize=True)
    )

    Q1 = np.einsum("bnef,afin->abie", H2[v, o, v, v], t2, optimize=True)
    Q2 = g[o, v, o, v] - 0.5 * np.einsum("mnie,bn->mbie", g[o, o, o, v], t1, optimize=True)
    Q2 = -np.einsum("mbie,am->abie", Q2, t1, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H2[v, v, o, v] = g[v, v, o, v] + Q1 + (
            - np.einsum("me,abim->abie", H1[o, v], t2, optimize=True)
            + np.einsum("abfe,fi->abie", H2[v, v, v, v], t1, optimize=True)
            + 0.5 * np.einsum("mnie,abmn->abie", g[o, o, o, v], t2, optimize=True)
    )

    return H1, H2
