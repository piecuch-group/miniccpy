import numpy as np

def build_hbar_rccsdt(T, f, g, o, v):
    """Calculate the one- and two-body components of the R-CCSDT 
    similarity-transformed Hamiltonian [H_N exp(T1+T2)]_C,
    defined by 
        H1[:, :] = < p | [H_N exp(T1+T2)]_C | q > 
        H2[:, :, :, :] = < pq | [H_N exp(T1+T2)]_C | rs >.
    """

    norbitals = f.shape[0]
    nunocc, nocc = f[v, o].shape
    n1 = nunocc * nocc

    t1, t2, t3 = T
    t2s = t2 - np.transpose(t2, (0, 1, 3, 2))
    # partially spin-summed t3(AbcIjk) = 2*t3(abcijk) - t3(abcjik) - t3(abckji)
    t3_s = (
            2.0 * t3
            - t3.transpose(0, 1, 2, 4, 3, 5)
            - t3.transpose(0, 1, 2, 5, 4, 3)
    )
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
                + np.einsum("nmfe,faenij->amij", g[o, o, v, v], t3_s, optimize=True)
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
                - np.einsum("nmfe,fabnim->abie", g[o, o, v, v], t3_s, optimize=True)
    )
    return H1, H2


def build_hbar_rccsd(T, f, g, o, v):
    """Calculate the one- and two-body components of the R-CCSD 
    similarity-transformed Hamiltonian [H_N exp(T1+T2)]_C,
    defined by 
        H1[:, :] = < p | [H_N exp(T1+T2)]_C | q > 
        H2[:, :, :, :] = < pq | [H_N exp(T1+T2)]_C | rs >.
    """

    norbitals = f.shape[0]
    nunocc, nocc = f[v, o].shape
    n1 = nunocc * nocc

    t1, t2 = T
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

def build_hbar_ccsd(T, f, g, o, v):
    """Calculate the one- and two-body components of the CCSD 
    similarity-transformed Hamiltonian [H_N exp(T1+T2)]_C,
    defined by 
        H1[:, :] = < p | [H_N exp(T1+T2)]_C | q > 
        H2[:, :, :, :] = < pq | [H_N exp(T1+T2)]_C | rs >.
    """

    norbitals = f.shape[0]
    nunocc, nocc = f[v, o].shape
    n1 = nunocc * nocc

    t1, t2 = T

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

    H2[o, o, v, v] = g[o, o, v, v].copy()

    return H1, H2

def build_hbar_ccsdt(T, f, g, o, v):
    """Calculate the one- and two-body components of the CCSDT 
    similarity-transformed Hamiltonian [H_N exp(T1+T2+T3)]_C,
    defined by 
        H1[:, :] = < p | [H_N exp(T1+T2+T3)]_C | q > 
        H2[:, :, :, :] = < pq | [H_N exp(T1+T2+T3)]_C | rs >.
    """

    norbitals = f.shape[0]
    nunocc, nocc = f[v, o].shape
    n1 = nunocc * nocc

    t1, t2, t3 = T

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
            + 0.5 * np.einsum("mnef,aefijn->amij", g[o, o, v, v], t3, optimize=True)
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
            - 0.5 * np.einsum("mnef,abfimn->abie", g[o, o, v, v], t3, optimize=True)
    )

    H2[o, o, v, v] = g[o, o, v, v].copy()

    return H1, H2

def build_hbar_cc3(T, f, g, o, v):
    """Calculate the one- and two-body components of the CCSDT 
    similarity-transformed Hamiltonian [H_N exp(T1+T2+T3)]_C,
    defined by 
        H1[:, :] = < p | [H_N exp(T1+T2+T3)]_C | q > 
        H2[:, :, :, :] = < pq | [H_N exp(T1+T2+T3)]_C | rs >.
    """
    from miniccpy.helper_cc3 import compute_cc3_intermediates

    norbitals = f.shape[0]
    nunocc, nocc = f[v, o].shape
    n1 = nunocc * nocc

    t1, t2 = T

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

    H2[o, o, v, v] = g[o, o, v, v].copy()

    # Parts contracted with T3
    eps = np.diagonal(f)
    n = np.newaxis
    e_abc = -eps[v, n, n] - eps[n, v, n] - eps[n, n, v]
    X_vooo, X_vvov = compute_cc3_intermediates(f, g, t1, t2, o, v)
    # premultiply by factor of 1/2 to compensate later antisymmetrization on the parts already built
    H2[v, v, o, v] *= 0.5
    H2[v, o, o, o] *= 0.5
    for i in range(nocc):
        for j in range(i + 1, nocc):
            for k in range(j + 1, nocc):
                # fock denominator for occupied
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                # -1/2 A(k/ij)A(abc) I(amij) * t(bcmk)
                t3_abc = -0.5 * np.einsum("am,bcm->abc", X_vooo[:, :, i, j], t2[:, :, :, k], optimize=True)
                t3_abc += 0.5 * np.einsum("am,bcm->abc", X_vooo[:, :, k, j], t2[:, :, :, i], optimize=True)
                t3_abc += 0.5 * np.einsum("am,bcm->abc", X_vooo[:, :, i, k], t2[:, :, :, j], optimize=True)
                # 1/2 A(i/jk)A(abc) I(abie) * t(ecjk)
                t3_abc += 0.5 * np.einsum("abe,ec->abc", X_vvov[:, :, i, :], t2[:, :, j, k], optimize=True)
                t3_abc -= 0.5 * np.einsum("abe,ec->abc", X_vvov[:, :, j, :], t2[:, :, i, k], optimize=True)
                t3_abc -= 0.5 * np.einsum("abe,ec->abc", X_vvov[:, :, k, :], t2[:, :, j, i], optimize=True)
                # Antisymmetrize A(abc)
                t3_abc -= np.transpose(t3_abc, (1, 0, 2)) + np.transpose(t3_abc, (2, 1, 0)) # A(a/bc)
                t3_abc -= np.transpose(t3_abc, (0, 2, 1)) # A(bc)
                # Divide t_abc by the denominator
                t3_abc /= (denom_occ + e_abc)
                # Compute diagram: A(i/jk) -g(jkef)*t3(abfijk)
                H2[v, v, o, v][:, :, i, :] -= 0.5 * np.einsum("ef,abf->abe", g[o, o, v, v][j, k, :, :], t3_abc, optimize=True)
                H2[v, v, o, v][:, :, j, :] += 0.5 * np.einsum("ef,abf->abe", g[o, o, v, v][i, k, :, :], t3_abc, optimize=True)
                H2[v, v, o, v][:, :, k, :] += 0.5 * np.einsum("ef,abf->abe", g[o, o, v, v][j, i, :, :], t3_abc, optimize=True)
                # Compute diagram: A(k/ij) g(:kef)*t3(aefijk)
                H2[v, o, o, o][:, :, i, j] += 0.5 * np.einsum("mef,aef->am", g[o, o, v, v][:, k, :, :], t3_abc, optimize=True)
                H2[v, o, o, o][:, :, j, k] += 0.5 * np.einsum("mef,aef->am", g[o, o, v, v][:, i, :, :], t3_abc, optimize=True)
                H2[v, o, o, o][:, :, i, k] -= 0.5 * np.einsum("mef,aef->am", g[o, o, v, v][:, j, :, :], t3_abc, optimize=True)

    # Antisymmetrize vvov and vooo parts
    H2[v, v, o, v] -= np.transpose(H2[v, v, o, v], (1, 0, 2, 3))
    H2[v, o, o, o] -= np.transpose(H2[v, o, o, o], (0, 1, 3, 2))
    # Manually clear all diagonal elements
    for a in range(nunocc):
        H2[v, v, o, v][a, a, :, :] *= 0.0
    for i in range(nocc):
        H2[v, o, o, o][:, :, i, i] *= 0.0

    return H1, H2

def build_hbar_ccsdta(T, f, g, o, v):
    """Calculate the one- and two-body components of the CCSDT
    similarity-transformed Hamiltonian [H_N exp(T1+T2+T3)]_C,
    defined by
        H1[:, :] = < p | [H_N exp(T1+T2+T3)]_C | q >
        H2[:, :, :, :] = < pq | [H_N exp(T1+T2+T3)]_C | rs >.
    """
    from miniccpy.energy import cc_energy

    t1, t2 = T
    nu, no = t1.shape
    norbitals = f.shape[0]

    # orbital denominators
    eps = np.diagonal(f)
    n = np.newaxis
    e_abc = -eps[v, n, n] - eps[n, v, n] - eps[n, n, v]
    e_abij = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
    e_ai = 1.0 / (-eps[v, n] + eps[n, o])

    # get old CC energy
    cc_energy_orig = cc_energy(t1, t2, f, g, o, v)

    # get residual containers for singles and doubles
    singles_res = np.zeros((nu, no))
    doubles_res = np.zeros((nu, nu, no, no))
    # build approximate T3 = <ijkabc|(V*T2)_C|0>/-D_MP(abcijk) in batches of abc
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                # fock denominator for occupied
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                # -1/2 A(k/ij)A(abc) I(amij) * t(bcmk)
                t3_abc = -0.5 * np.einsum("am,bcm->abc", g[v, o, o, o][:, :, i, j], t2[:, :, :, k], optimize=True)
                t3_abc += 0.5 * np.einsum("am,bcm->abc", g[v, o, o, o][:, :, k, j], t2[:, :, :, i], optimize=True)
                t3_abc += 0.5 * np.einsum("am,bcm->abc", g[v, o, o, o][:, :, i, k], t2[:, :, :, j], optimize=True)
                # 1/2 A(i/jk)A(abc) I(abie) * t(ecjk)
                t3_abc += 0.5 * np.einsum("abe,ec->abc", g[v, v, o, v][:, :, i, :], t2[:, :, j, k], optimize=True)
                t3_abc -= 0.5 * np.einsum("abe,ec->abc", g[v, v, o, v][:, :, j, :], t2[:, :, i, k], optimize=True)
                t3_abc -= 0.5 * np.einsum("abe,ec->abc", g[v, v, o, v][:, :, k, :], t2[:, :, j, i], optimize=True)
                # Antisymmetrize A(abc)
                t3_abc -= np.transpose(t3_abc, (1, 0, 2)) + np.transpose(t3_abc, (2, 1, 0)) # A(a/bc)
                t3_abc -= np.transpose(t3_abc, (0, 2, 1)) # A(bc)
                # Divide t_abc by the denominator
                t3_abc /= (denom_occ + e_abc)
                # Compute diagram: 1/2 A(i/jk) v(jkbc) * t(abcijk)
                singles_res[:, i] += 0.5 * np.einsum("bc,abc->a", g[o, o, v, v][j, k, :, :], t3_abc, optimize=True)
                singles_res[:, j] -= 0.5 * np.einsum("bc,abc->a", g[o, o, v, v][i, k, :, :], t3_abc, optimize=True)
                singles_res[:, k] -= 0.5 * np.einsum("bc,abc->a", g[o, o, v, v][j, i, :, :], t3_abc, optimize=True)
                # Compute diagram: A(ij) [A(k/ij) h(ke) * t3(abeijk)]
                doubles_res[:, :, i, j] += 0.5 * np.einsum("e,abe->ab", f[o, v][k, :], t3_abc, optimize=True)  # (1)
                doubles_res[:, :, j, k] += 0.5 * np.einsum("e,abe->ab", f[o, v][i, :], t3_abc, optimize=True)  # (ik)
                doubles_res[:, :, i, k] -= 0.5 * np.einsum("e,abe->ab", f[o, v][j, :], t3_abc, optimize=True)  # (jk)
                # Compute diagram: -A(j/ik) h(ik:f) * t3(abfijk)
                doubles_res[:, :, :, j] -= 0.5 * np.einsum("mf,abf->abm", g[o, o, o, v][i, k, :, :], t3_abc, optimize=True)
                doubles_res[:, :, :, i] += 0.5 * np.einsum("mf,abf->abm", g[o, o, o, v][j, k, :, :], t3_abc, optimize=True)
                doubles_res[:, :, :, k] += 0.5 * np.einsum("mf,abf->abm", g[o, o, o, v][i, j, :, :], t3_abc, optimize=True)
                # Compute diagram: 1/2 A(k/ij) h(akef) * t3(ebfijk)
                doubles_res[:, :, i, j] += 0.5 * np.einsum("aef,ebf->ab", g[v, o, v, v][:, k, :, :], t3_abc, optimize=True)
                doubles_res[:, :, j, k] += 0.5 * np.einsum("aef,ebf->ab", g[v, o, v, v][:, i, :, :], t3_abc, optimize=True)
                doubles_res[:, :, i, k] -= 0.5 * np.einsum("aef,ebf->ab", g[v, o, v, v][:, j, :, :], t3_abc, optimize=True)
    # Antisymmetrize
    doubles_res -= np.transpose(doubles_res, (1, 0, 2, 3))
    doubles_res -= np.transpose(doubles_res, (0, 1, 3, 2))
    # Manually clear all diagonal elements
    for a in range(nu):
        doubles_res[a, a, :, :] *= 0.0
    for i in range(no):
        doubles_res[:, :, i, i] *= 0.0

    # update CCSD amplitudes with contributions from T3
    t1 += singles_res * e_ai
    t2 += doubles_res * e_abij

    # compute new CC energy
    cc_energy_new = cc_energy(t1, t2, f, g, o, v)

    print(f"    CCSD Correlation Energy:   {cc_energy_orig}")
    print(f"    CCSDT(a) Correlation Energy:    {cc_energy_new}\n")

    # Allocate H1 and H2 arrays
    H1 = np.zeros((norbitals, norbitals))
    H2 = np.zeros((norbitals, norbitals, norbitals, norbitals))
    
    # Now that T1/T2 are updated, compute the additional T3 contribution to h(vooo) and h(vvov)
    h_t3_vvov = np.zeros((nu, nu, no, nu))
    h_t3_vooo = np.zeros((nu, no, no, no))
    # build approximate T3 = <ijkabc|(V*T2)_C|0>/-D_MP(abcijk) in batches of abc
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                # fock denominator for occupied
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                # -1/2 A(k/ij)A(abc) I(amij) * t(bcmk)
                t3_abc = -0.5 * np.einsum("am,bcm->abc", g[v, o, o, o][:, :, i, j], t2[:, :, :, k], optimize=True)
                t3_abc += 0.5 * np.einsum("am,bcm->abc", g[v, o, o, o][:, :, k, j], t2[:, :, :, i], optimize=True)
                t3_abc += 0.5 * np.einsum("am,bcm->abc", g[v, o, o, o][:, :, i, k], t2[:, :, :, j], optimize=True)
                # 1/2 A(i/jk)A(abc) I(abie) * t(ecjk)
                t3_abc += 0.5 * np.einsum("abe,ec->abc", g[v, v, o, v][:, :, i, :], t2[:, :, j, k], optimize=True)
                t3_abc -= 0.5 * np.einsum("abe,ec->abc", g[v, v, o, v][:, :, j, :], t2[:, :, i, k], optimize=True)
                t3_abc -= 0.5 * np.einsum("abe,ec->abc", g[v, v, o, v][:, :, k, :], t2[:, :, j, i], optimize=True)
                # Antisymmetrize A(abc)
                t3_abc -= np.transpose(t3_abc, (1, 0, 2)) + np.transpose(t3_abc, (2, 1, 0)) # A(a/bc)
                t3_abc -= np.transpose(t3_abc, (0, 2, 1)) # A(bc)
                # Divide t_abc by the denominator
                t3_abc /= (denom_occ + e_abc)
                # Compute diagram: A(i/jk) -g(jkef)*t3(abfijk)
                h_t3_vvov[:, :, i, :] -= 0.5 * np.einsum("ef,abf->abe", g[o, o, v, v][j, k, :, :], t3_abc, optimize=True)
                h_t3_vvov[:, :, j, :] += 0.5 * np.einsum("ef,abf->abe", g[o, o, v, v][i, k, :, :], t3_abc, optimize=True)
                h_t3_vvov[:, :, k, :] += 0.5 * np.einsum("ef,abf->abe", g[o, o, v, v][j, i, :, :], t3_abc, optimize=True)
                # Compute diagram: A(k/ij) g(:kef)*t3(aefijk)
                h_t3_vooo[:, :, i, j] += 0.5 * np.einsum("mef,aef->am", g[o, o, v, v][:, k, :, :], t3_abc, optimize=True)
                h_t3_vooo[:, :, j, k] += 0.5 * np.einsum("mef,aef->am", g[o, o, v, v][:, i, :, :], t3_abc, optimize=True)
                h_t3_vooo[:, :, i, k] -= 0.5 * np.einsum("mef,aef->am", g[o, o, v, v][:, j, :, :], t3_abc, optimize=True)
    # Antisymmetrize
    h_t3_vvov -= np.transpose(h_t3_vvov, (1, 0, 2, 3))
    h_t3_vooo -= np.transpose(h_t3_vooo, (0, 1, 3, 2))
    # Manually clear all diagonal elements
    for a in range(nu):
        h_t3_vvov[a, a, :, :] *= 0.0
    for i in range(no):
        h_t3_vooo[:, :, i, i] *= 0.0

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
    H2[v, o, o, o] += h_t3_vooo # add in T3 contribution

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
    H2[v, v, o, v] += h_t3_vvov # add in T3 contribution

    H2[o, o, v, v] = g[o, o, v, v].copy()

    return (t1, t2), (H1, H2)
