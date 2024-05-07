import numpy as np

def compute_r3(r1, r2, t1, t2, omega, f, g, o, v):

    # CCS Hbar elements
    h_voov = g[v, o, o, v] + (
            -np.einsum("nmie,an->amie", g[o, o, o, v], t1, optimize=True)
            +np.einsum("amfe,fi->amie", g[v, o, v, v], t1, optimize=True)
            -np.einsum("mnef,an,fi->amie", g[o, o, v, v], t1, t1, optimize=True)
    )
    h_oooo = 0.5 * g[o, o, o, o] + (
            +np.einsum("nmjf,fi->mnij", g[o, o, o, v], t1, optimize=True)
            +0.5 * np.einsum("mnef,ei,fj->mnij", g[o, o, v, v], t1, t1, optimize=True)
    )
    h_oooo -= np.transpose(h_oooo, (0, 1, 3, 2))
    h_vvvv = 0.5 * g[v, v, v, v] + (
            -np.einsum("anef,bn->abef", g[v, o, v, v], t1, optimize=True)
            +0.5 * np.einsum("mnef,am,bn->abef", g[o, o, v, v,], t1, t1, optimize=True)
    )
    h_vvvv -= np.transpose(h_vvvv, (1, 0, 2, 3))
    h_vooo = 0.5 * g[v, o, o, o] + (
            +np.einsum("amie,ej->amij", g[v, o, o, v], t1, optimize=True)
            -0.5 * np.einsum("nmij,an->amij", g[o, o, o, o], t1, optimize=True)
            -np.einsum("mnjf,an,fi->amij", g[o, o, o, v], t1, t1, optimize=True)
            +0.5 * np.einsum("amef,ei,fj->amij", g[v, o, v, v], t1, t1, optimize=True)
            -0.5 * np.einsum("mnef,an,fi,ej->amij", g[o, o, v, v], t1, t1, t1, optimize=True)
    )
    h_vooo -= np.transpose(h_vooo, (0, 1, 3, 2))
    # Added in for ROHF
    h_vooo += np.einsum("me,aeij->amij", f[o, v], t2, optimize=True)
    h_vvov = 0.5 * g[v, v, o, v] + (
            -np.einsum("anie,bn->abie", g[v, o, o, v], t1, optimize=True)
            +0.5 * np.einsum("abfe,fi->abie", g[v, v, v, v], t1, optimize=True)
            -np.einsum("bnef,an,fi->abie", g[v, o, v, v], t1, t1, optimize=True)
            +0.5 * np.einsum("mnie,bn,am->abie", g[o, o, o, v], t1, t1, optimize=True)
            +0.5 * np.einsum("mnef,bm,an,fi->abie", g[o, o, v, v], t1, t1, t1, optimize=True)
    )
    h_vvov -= np.transpose(h_vvov, (1, 0, 2, 3))

    X_vooo = (
                np.einsum("amie,ej->amij", h_voov, r1, optimize=True)
               -0.5 * np.einsum("mnji,an->amij", h_oooo, r1, optimize=True)
    )
    X_vooo -= np.transpose(X_vooo, (0, 1, 3, 2))
    X_vvov = (
                -np.einsum("amie,bm->abie", h_voov, r1, optimize=True)
                +0.5 * np.einsum("abfe,fi->abie", h_vvvv, r1, optimize=True)
    )
    X_vvov -= np.transpose(X_vvov, (1, 0, 2, 3))

    # <ijkabc| [H(1)*T2*R1]_C | 0 >
    X3 = 0.25 * np.einsum("baje,ecik->abcijk", X_vvov, t2, optimize=True)
    X3 -= 0.25 * np.einsum("bmji,acmk->abcijk", X_vooo, t2, optimize=True)
    # <ijkabc| [H(1)*R2]_C | 0 >
    X3 += 0.25 * np.einsum("baje,ecik->abcijk", h_vvov, r2, optimize=True)
    X3 -= 0.25 * np.einsum("bmji,acmk->abcijk", h_vooo, r2, optimize=True)
    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    X3 -= np.transpose(X3, (0, 1, 2, 3, 5, 4))
    X3 -= np.transpose(X3, (0, 1, 2, 4, 3, 5)) + np.transpose(X3, (0, 1, 2, 5, 4, 3))
    X3 -= np.transpose(X3, (0, 2, 1, 3, 4, 5))
    X3 -= np.transpose(X3, (1, 0, 2, 3, 4, 5)) + np.transpose(X3, (2, 1, 0, 3, 4, 5))

    eps = np.diagonal(f)
    n = np.newaxis
    e_abcijk = (-eps[v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                    + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o])
    return X3 / (omega + e_abcijk)

def compute_t3(t1, t2, f, g, o, v):
    """Compute the projection of the CCSDT Hamiltonian on triples
        X[a, b, c, i, j, k] = < ijkabc | (H_N exp(T1+T2+T3))_C | 0 >
    """

    # h(vvov) and h(vooo) intermediates resulting from exp(-T1) H_N exp(T1)
    Q1 = -np.einsum("mnfe,an->amef", g[o, o, v, v], t1, optimize=True)
    I_vovv = g[v, o, v, v] + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    I_ooov = g[o, o, o, v] + 0.5 * Q1

    Q1 = -np.einsum("bmfe,am->abef", I_vovv, t1, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    I_vvvv = g[v, v, v, v] + Q1

    Q1 = +np.einsum("nmje,ei->mnij", I_ooov, t1, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    I_oooo = g[o, o, o, o] + Q1

    Q1 = g[v, o, o, v] + 0.5 * np.einsum("amef,ei->amif", g[v, o, v, v], t1, optimize=True)
    Q1 = np.einsum("amif,fj->amij", Q1, t1, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    I_vooo = g[v, o, o, o] + Q1 - np.einsum("nmij,an->amij", I_oooo, t1, optimize=True)
    # Added in for ROHF
    I_vooo += np.einsum("me,aeij->amij", f[o, v], t2, optimize=True)

    Q1 = g[o, v, o, v] - 0.5 * np.einsum("mnie,bn->mbie", g[o, o, o, v], t1, optimize=True)
    Q1 = -np.einsum("mbie,am->abie", Q1, t1, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    I_vvov = g[v, v, o, v] + Q1 + np.einsum("abfe,fi->abie", I_vvvv, t1, optimize=True)
    # Added in for ROHF
    #I_vvov -= np.einsum("me,abim->abie", f[o, v], t2, optimize=True)

    eps = np.diagonal(f)
    n = np.newaxis
    e_abcijk = 1.0 / (- eps[v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                    + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o] + energy_shift )

    triples_res = -0.25 * np.einsum("amij,bcmk->abcijk", I_vooo, t2, optimize=True)
    triples_res += 0.25 * np.einsum("abie,ecjk->abcijk", I_vvov, t2, optimize=True)

    triples_res -= np.transpose(triples_res, (0, 1, 2, 3, 5, 4)) # (jk)
    triples_res -= np.transpose(triples_res, (0, 1, 2, 4, 3, 5)) + np.transpose(triples_res, (0, 1, 2, 5, 4, 3)) # (i/jk)
    triples_res -= np.transpose(triples_res, (0, 2, 1, 3, 4, 5)) # (bc)
    triples_res -= np.transpose(triples_res, (2, 1, 0, 3, 4, 5)) + np.transpose(triples_res, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return triples_res * e_abcijk

def compute_l3(l1, l2, t1, t2, omega, f, g, o, v):

    # < 0 | L1 * H(2) | ijkabc >
    l3 = (9.0 / 36.0) * np.einsum("ai,jkbc->abcijk", l1, g[o, o, v, v], optimize=True)

    # < 0 | L2 * H(2) | ijkabc >
    l3 += (9.0 / 36.0) * np.einsum("bcjk,ia->abcijk", l2, f[o, v], optimize=True)

    l3 += (9.0 / 36.0) * np.einsum("ebij,ekac->abcijk", l2, g[v, o, v, v], optimize=True)
    l3 -= (9.0 / 36.0) * np.einsum("abmj,ikmc->abcijk", l2, g[o, o, o, v], optimize=True)
    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    l3 -= np.transpose(l3, (0, 1, 2, 3, 5, 4))
    l3 -= np.transpose(l3, (0, 1, 2, 4, 3, 5)) + np.transpose(l3, (0, 1, 2, 5, 4, 3))
    l3 -= np.transpose(l3, (0, 2, 1, 3, 4, 5))
    l3 -= np.transpose(l3, (1, 0, 2, 3, 4, 5)) + np.transpose(l3, (2, 1, 0, 3, 4, 5))

    eps = np.diagonal(f)
    n = np.newaxis
    e_abcijk = (-eps[v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                    + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o])
    return l3 / (omega + e_abcijk)

def compute_cc3_intermediates(f, g, t1, t2, o, v):

    # h(vvov) and h(vooo) intermediates resulting from exp(-T1) H_N exp(T1)
    Q1 = -np.einsum("mnfe,an->amef", g[o, o, v, v], t1, optimize=True)
    I_vovv = g[v, o, v, v] + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    I_ooov = g[o, o, o, v] + 0.5 * Q1

    Q1 = -np.einsum("bmfe,am->abef", I_vovv, t1, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    I_vvvv = g[v, v, v, v] + Q1

    Q1 = +np.einsum("nmje,ei->mnij", I_ooov, t1, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    I_oooo = g[o, o, o, o] + Q1

    Q1 = g[v, o, o, v] + 0.5 * np.einsum("amef,ei->amif", g[v, o, v, v], t1, optimize=True)
    Q1 = np.einsum("amif,fj->amij", Q1, t1, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    I_vooo = g[v, o, o, o] + Q1 - np.einsum("nmij,an->amij", I_oooo, t1, optimize=True)
    # Added in for ROHF
    I_vooo += np.einsum("me,aeij->amij", f[o, v], t2, optimize=True)

    Q1 = g[o, v, o, v] - 0.5 * np.einsum("mnie,bn->mbie", g[o, o, o, v], t1, optimize=True)
    Q1 = -np.einsum("mbie,am->abie", Q1, t1, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    I_vvov = g[v, v, o, v] + Q1 + np.einsum("abfe,fi->abie", I_vvvv, t1, optimize=True)
    # Added in for ROHF
    #I_vvov -= np.einsum("me,abim->abie", f[o, v], t2, optimize=True)

    return I_vooo, I_vvov

def compute_eomcc3_intermediates(r1, r2, t1, t2, f, g, o, v):
    # CCS Hbar elements
    h_voov = g[v, o, o, v] + (
            -np.einsum("nmie,an->amie", g[o, o, o, v], t1, optimize=True)
            +np.einsum("amfe,fi->amie", g[v, o, v, v], t1, optimize=True)
            -np.einsum("mnef,an,fi->amie", g[o, o, v, v], t1, t1, optimize=True)
    )
    h_oooo = 0.5 * g[o, o, o, o] + (
            +np.einsum("nmjf,fi->mnij", g[o, o, o, v], t1, optimize=True)
            +0.5 * np.einsum("mnef,ei,fj->mnij", g[o, o, v, v], t1, t1, optimize=True)
    )
    h_oooo -= np.transpose(h_oooo, (0, 1, 3, 2))
    h_vvvv = 0.5 * g[v, v, v, v] + (
            -np.einsum("anef,bn->abef", g[v, o, v, v], t1, optimize=True)
            +0.5 * np.einsum("mnef,am,bn->abef", g[o, o, v, v,], t1, t1, optimize=True)
    )
    h_vvvv -= np.transpose(h_vvvv, (1, 0, 2, 3))
    h_vooo = 0.5 * g[v, o, o, o] + (
            +np.einsum("amie,ej->amij", g[v, o, o, v], t1, optimize=True)
            -0.5 * np.einsum("nmij,an->amij", g[o, o, o, o], t1, optimize=True)
            -np.einsum("mnjf,an,fi->amij", g[o, o, o, v], t1, t1, optimize=True)
            +0.5 * np.einsum("amef,ei,fj->amij", g[v, o, v, v], t1, t1, optimize=True)
            -0.5 * np.einsum("mnef,an,fi,ej->amij", g[o, o, v, v], t1, t1, t1, optimize=True)
    )
    h_vooo -= np.transpose(h_vooo, (0, 1, 3, 2))
    # Added in for ROHF
    h_vooo += np.einsum("me,aeij->amij", f[o, v], t2, optimize=True)
    h_vvov = 0.5 * g[v, v, o, v] + (
            -np.einsum("anie,bn->abie", g[v, o, o, v], t1, optimize=True)
            +0.5 * np.einsum("abfe,fi->abie", g[v, v, v, v], t1, optimize=True)
            -np.einsum("bnef,an,fi->abie", g[v, o, v, v], t1, t1, optimize=True)
            +0.5 * np.einsum("mnie,bn,am->abie", g[o, o, o, v], t1, t1, optimize=True)
            +0.5 * np.einsum("mnef,bm,an,fi->abie", g[o, o, v, v], t1, t1, t1, optimize=True)
    )
    h_vvov -= np.transpose(h_vvov, (1, 0, 2, 3))

    X_vooo = (
                np.einsum("amie,ej->amij", h_voov, r1, optimize=True)
               -0.5 * np.einsum("mnji,an->amij", h_oooo, r1, optimize=True)
    )
    X_vooo -= np.transpose(X_vooo, (0, 1, 3, 2))
    X_vvov = (
                -np.einsum("amie,bm->abie", h_voov, r1, optimize=True)
                +0.5 * np.einsum("abfe,fi->abie", h_vvvv, r1, optimize=True)
    )
    X_vvov -= np.transpose(X_vvov, (1, 0, 2, 3))
    return h_vvov, h_vooo, X_vvov, X_vooo

def compute_leftcc3_intermediates(t1, t2, f, g, o, v):
    # CCS Hbar elements
    h_voov = g[v, o, o, v] + (
            -np.einsum("nmie,an->amie", g[o, o, o, v], t1, optimize=True)
            +np.einsum("amfe,fi->amie", g[v, o, v, v], t1, optimize=True)
            -np.einsum("mnef,an,fi->amie", g[o, o, v, v], t1, t1, optimize=True)
    )
    h_oooo = 0.5 * g[o, o, o, o] + (
            +np.einsum("nmjf,fi->mnij", g[o, o, o, v], t1, optimize=True)
            +0.5 * np.einsum("mnef,ei,fj->mnij", g[o, o, v, v], t1, t1, optimize=True)
    )
    h_oooo -= np.transpose(h_oooo, (0, 1, 3, 2))
    h_vvvv = 0.5 * g[v, v, v, v] + (
            -np.einsum("anef,bn->abef", g[v, o, v, v], t1, optimize=True)
            +0.5 * np.einsum("mnef,am,bn->abef", g[o, o, v, v,], t1, t1, optimize=True)
    )
    h_vvvv -= np.transpose(h_vvvv, (1, 0, 2, 3))
    h_vooo = 0.5 * g[v, o, o, o] + (
            +np.einsum("amie,ej->amij", g[v, o, o, v], t1, optimize=True)
            -0.5 * np.einsum("nmij,an->amij", g[o, o, o, o], t1, optimize=True)
            -np.einsum("mnjf,an,fi->amij", g[o, o, o, v], t1, t1, optimize=True)
            +0.5 * np.einsum("amef,ei,fj->amij", g[v, o, v, v], t1, t1, optimize=True)
            -0.5 * np.einsum("mnef,an,fi,ej->amij", g[o, o, v, v], t1, t1, t1, optimize=True)
    )
    h_vooo -= np.transpose(h_vooo, (0, 1, 3, 2))
    # Added in for ROHF
    h_vooo += np.einsum("me,aeij->amij", f[o, v], t2, optimize=True)
    h_vvov = 0.5 * g[v, v, o, v] + (
            -np.einsum("anie,bn->abie", g[v, o, o, v], t1, optimize=True)
            +0.5 * np.einsum("abfe,fi->abie", g[v, v, v, v], t1, optimize=True)
            -np.einsum("bnef,an,fi->abie", g[v, o, v, v], t1, t1, optimize=True)
            +0.5 * np.einsum("mnie,bn,am->abie", g[o, o, o, v], t1, t1, optimize=True)
            +0.5 * np.einsum("mnef,bm,an,fi->abie", g[o, o, v, v], t1, t1, t1, optimize=True)
    )
    h_vvov -= np.transpose(h_vvov, (1, 0, 2, 3))
    return h_vvov, h_vooo, h_voov, h_vvvv, h_oooo

def get_lr_intermediates(l1, l2, t2, f, H1, H2, h_vvov, h_vooo, omega, e_abc, o, v):
    nu, _, no, _ = t2.shape
    X1 = {"vo": []}
    X2 = {"ooov": [], "vovv": []}
    # (L2 * T3)_C
    #X1["vo"] = 0.25 * np.einsum("efmn,aefimn->ai", l2, t3, optimize=True)
    X1["vo"] = np.zeros((nu, no))
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                # fock denominator for occupied
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                # -1/2 A(k/ij)A(abc) I(amij) * t(bcmk)
                t3_abc = -0.5 * np.einsum("am,bcm->abc", h_vooo[:, :, i, j], t2[:, :, :, k], optimize=True)
                t3_abc += 0.5 * np.einsum("am,bcm->abc", h_vooo[:, :, k, j], t2[:, :, :, i], optimize=True)
                t3_abc += 0.5 * np.einsum("am,bcm->abc", h_vooo[:, :, i, k], t2[:, :, :, j], optimize=True)
                # 1/2 A(i/jk)A(abc) I(abie) * t(ecjk)
                t3_abc += 0.5 * np.einsum("abe,ec->abc", h_vvov[:, :, i, :], t2[:, :, j, k], optimize=True)
                t3_abc -= 0.5 * np.einsum("abe,ec->abc", h_vvov[:, :, j, :], t2[:, :, i, k], optimize=True)
                t3_abc -= 0.5 * np.einsum("abe,ec->abc", h_vvov[:, :, k, :], t2[:, :, j, i], optimize=True)
                # Antisymmetrize A(abc)
                t3_abc -= np.transpose(t3_abc, (1, 0, 2)) + np.transpose(t3_abc, (2, 1, 0)) # A(a/bc)
                t3_abc -= np.transpose(t3_abc, (0, 2, 1)) # A(bc)
                # Divide t_abc by the denominator
                t3_abc /= (denom_occ + e_abc)
                # 1/4 A(i/jk) l2(bcjk) * t3(abcijk)
                X1["vo"][:, i] += 0.5 * np.einsum("bc,abc->a", l2[:, :, j, k], t3_abc, optimize=True)
                X1["vo"][:, j] -= 0.5 * np.einsum("bc,abc->a", l2[:, :, i, k], t3_abc, optimize=True)
                X1["vo"][:, k] -= 0.5 * np.einsum("bc,abc->a", l2[:, :, j, i], t3_abc, optimize=True)
    # (L3 * T2)_C
    #X2["ooov"] = 0.5 * np.einsum('aefijn,efmn->jima', l3, t2, optimize=True)
    X2["ooov"] = np.zeros((no, no, no, nu))
    #X2["vovv"] = -0.5 * np.einsum('abfimn,efmn->eiba', l3, t2, optimize=True)
    X2["vovv"] = np.zeros((nu, no, nu, nu))
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                # fock denominator for occupied
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                l3_abc = 0.5 * (
                        np.einsum("eba,ec->abc", H2[v, o, v, v][:, i, :, :], l2[:, :, j, k], optimize=True)
                        -np.einsum("eba,ec->abc", H2[v, o, v, v][:, j, :, :], l2[:, :, i, k], optimize=True)
                        -np.einsum("eba,ec->abc", H2[v, o, v, v][:, k, :, :], l2[:, :, j, i], optimize=True)
                )
                l3_abc -= 0.5 * (
                        np.einsum("ma,bcm->abc", H2[o, o, o, v][j, i, :, :], l2[:, :, :, k], optimize=True)
                        -np.einsum("ma,bcm->abc", H2[o, o, o, v][k, i, :, :], l2[:, :, :, j], optimize=True)
                        -np.einsum("ma,bcm->abc", H2[o, o, o, v][j, k, :, :], l2[:, :, :, i], optimize=True)
                )
                l3_abc += 0.5 * (
                        np.einsum("ab,c->abc", H2[o, o, v, v][i, j, :, :], l1[:, k], optimize=True)
                        -np.einsum("ab,c->abc", H2[o, o, v, v][k, j, :, :], l1[:, i], optimize=True)
                        -np.einsum("ab,c->abc", H2[o, o, v, v][i, k, :, :], l1[:, j], optimize=True)
                )
                l3_abc += 0.5 * (
                        np.einsum("a,bc->abc", H1[o, v][i, :], l2[:, :, j, k], optimize=True)
                        -np.einsum("a,bc->abc", H1[o, v][j, :], l2[:, :, i, k], optimize=True)
                        -np.einsum("a,bc->abc", H1[o, v][k, :], l2[:, :, j, i], optimize=True)
                )
                # antisymmetrize A(abc)
                l3_abc -= np.transpose(l3_abc, (1, 0, 2)) + np.transpose(l3_abc, (2, 1, 0)) # (a/bc)
                l3_abc -= np.transpose(l3_abc, (0, 2, 1)) # (bc)
                # Divide l_abc by the denominator
                l3_abc /= (omega + denom_occ + e_abc)
                # X2(jima) = 1/2 A(k/ij) l3(aefijk) * t2(efmk)
                X2["ooov"][i, j, :, :] -= 0.5 * np.einsum("aef,efm->ma", l3_abc, t2[:, :, :, k], optimize=True)
                X2["ooov"][j, k, :, :] -= 0.5 * np.einsum("aef,efm->ma", l3_abc, t2[:, :, :, i], optimize=True)
                X2["ooov"][i, k, :, :] += 0.5 * np.einsum("aef,efm->ma", l3_abc, t2[:, :, :, j], optimize=True)
                # X2(eiba) = -1/2 A(i/jk) l3(abfijk) * t2(efjk)
                X2["vovv"][:, i, :, :] -= np.einsum("abf,ef->eba", l3_abc, t2[:, :, j, k], optimize=True)
                X2["vovv"][:, j, :, :] += np.einsum("abf,ef->eba", l3_abc, t2[:, :, i, k], optimize=True)
                X2["vovv"][:, k, :, :] += np.einsum("abf,ef->eba", l3_abc, t2[:, :, j, i], optimize=True)
    # antisymmetrize 
    for i in range(no):
        for j in range(i + 1, no):
            X2["ooov"][j, i, :, :] = -X2["ooov"][i, j, :, :]
    # clear diagonal elements
    for a in range(nu):
        X2["vovv"][:, :, a, a] *= 0.0
    return X1, X2

