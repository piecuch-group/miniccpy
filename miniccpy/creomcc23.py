import numpy as np
#from numba import njit
from miniccpy.hbar_diagonal import get_3body_hbar_triples_diagonal

def kernel(T, R, L, r0, omega, fock, H1, H2, o, v):

    t1, t2 = T
    r1, r2 = R
    l1, l2 = L
    # orbital dimensions
    nu, no = t1.shape
    # get 3-body Hbar triples diagonal
    d3v, d3o = get_3body_hbar_triples_diagonal(H2[o, o, v, v], t2)

    # Intermediates
    I_vooo = H2[v, o, o, o] - np.einsum("me,aeij->amij", H1[o, v], t2, optimize=True)

    X_ov = np.einsum("mnef,fn->me", H2[o, o, v, v], r1, optimize=True)
    X_vvov =(
        np.einsum("amje,bm->baje", H2[v, o, o, v], r1, optimize=True)
        + np.einsum("amfe,bejm->bajf", H2[v, o, v, v], r2, optimize=True)
        + 0.5 * np.einsum("abfe,ej->bajf", H2[v, v, v, v], r1, optimize=True)
        + 0.25 * np.einsum("nmje,abmn->baje", H2[o, o, o, v], r2, optimize=True)
        - 0.5 * np.einsum("me,abmj->baje", X_ov, t2, optimize=True) 
    )
    X_vvov -= np.transpose(X_vvov, (1, 0, 2, 3))

    X_vooo = (
        -np.einsum("bmie,ej->bmji", H2[v, o, o, v], r1, optimize=True)
        +np.einsum("nmie,bejm->bnji", H2[o, o, o, v], r2, optimize=True)
        - 0.5 * np.einsum("nmij,bm->bnji", H2[o, o, o, o], r1, optimize=True)
        + 0.25 * np.einsum("bmfe,efij->bmji", H2[v, o, v, v], r2, optimize=True)
    )
    X_vooo -= np.transpose(X_vooo, (0, 1, 3, 2))

    # Ground-state moment
    #M3 = r0 * (9.0 / 36.0) * (
    #         np.einsum("abie,ecjk->abcijk", H2[v, v, o, v], t2, optimize=True)
    #        -np.einsum("amij,bcmk->abcijk", I_vooo, t2, optimize=True)
    #)
    # Excited-state moment
    #M3 += 0.25 * np.einsum("baje,ecik->abcijk", X_vvov, t2, optimize=True)
    #M3 += 0.25 * np.einsum("baje,ecik->abcijk", H2[v, v, o, v], r2, optimize=True)
    #M3 -= 0.25 * np.einsum("bmji,acmk->abcijk", X_vooo, t2, optimize=True)
    #M3 -= 0.25 * np.einsum("bmji,acmk->abcijk", H2[v, o, o, o], r2, optimize=True)
    #M3 -= np.transpose(M3, (0, 1, 2, 3, 5, 4)) # (jk)
    #M3 -= np.transpose(M3, (0, 1, 2, 4, 3, 5)) + np.transpose(M3, (0, 1, 2, 5, 4, 3)) # (i/jk)
    #M3 -= np.transpose(M3, (0, 2, 1, 3, 4, 5)) # (bc)
    #M3 -= np.transpose(M3, (2, 1, 0, 3, 4, 5)) + np.transpose(M3, (1, 0, 2, 3, 4, 5)) # (a/bc)

    # Left 
    #L3 = (9.0 / 36.0) * (
    #        np.einsum("ijab,ck->abcijk", H2[o, o, v, v], l1, optimize=True)
    #       +np.einsum("ia,bcjk->abcijk", H1[o, v], l2, optimize=True)
    #       +np.einsum("eiba,ecjk->abcijk", H2[v, o, v, v], l2, optimize=True)
    #       -np.einsum("jima,bcmk->abcijk", H2[o, o, o, v], l2, optimize=True)
    #)
    #L3 -= np.transpose(L3, (0, 1, 2, 3, 5, 4)) # (jk)
    #L3 -= np.transpose(L3, (0, 1, 2, 4, 3, 5)) + np.transpose(L3, (0, 1, 2, 5, 4, 3)) # (i/jk)
    #L3 -= np.transpose(L3, (0, 2, 1, 3, 4, 5)) # (bc)
    #L3 -= np.transpose(L3, (2, 1, 0, 3, 4, 5)) + np.transpose(L3, (1, 0, 2, 3, 4, 5)) # (a/bc)

    delta_A, delta_B, delta_C, delta_D = correction_in_loop(t1, t2, l1, l2, r1, r2, r0, omega, no, nu, fock, H1, H2, I_vooo, X_vooo, X_vvov, d3o, d3v, o, v)

    # Store triples corrections in dictionary
    delta_T = {"A": delta_A, "B": delta_B, "C": delta_C, "D": delta_D}
    return delta_T

def moments_ijk(i, j, k, I_vooo, h2_vooo, h2_vvov, x2_vooo, x2_vvov, t2, r2, r0):
    '''Computes the moment M(abc) = <ijkabc|H(2)|0> for a fixed i,j,k for i<j<k.'''
    M3 = r0 * 0.5 * (
            np.einsum("abe,ec->abc", h2_vvov[:, :, i, :], t2[:, :, j, k], optimize=True)
            -np.einsum("abe,ec->abc", h2_vvov[:, :, j, :], t2[:, :, i, k], optimize=True)
            -np.einsum("abe,ec->abc", h2_vvov[:, :, k, :], t2[:, :, j, i], optimize=True)
    )
    M3 -= r0 * 0.5 * (
            np.einsum("am,bcm->abc", I_vooo[:, :, i, j], t2[:, :, :, k], optimize=True)
            -np.einsum("am,bcm->abc", I_vooo[:, :, k, j], t2[:, :, :, i], optimize=True)
            -np.einsum("am,bcm->abc", I_vooo[:, :, i, k], t2[:, :, :, j], optimize=True)
    )
    M3 += 0.5 * (
            np.einsum("abe,ec->abc", h2_vvov[:, :, i, :], r2[:, :, j, k], optimize=True)
            -np.einsum("abe,ec->abc", h2_vvov[:, :, j, :], r2[:, :, i, k], optimize=True)
            -np.einsum("abe,ec->abc", h2_vvov[:, :, k, :], r2[:, :, j, i], optimize=True)
    )
    M3 -= 0.5 * (
            np.einsum("am,bcm->abc", h2_vooo[:, :, i, j], r2[:, :, :, k], optimize=True)
            -np.einsum("am,bcm->abc", h2_vooo[:, :, k, j], r2[:, :, :, i], optimize=True)
            -np.einsum("am,bcm->abc", h2_vooo[:, :, i, k], r2[:, :, :, j], optimize=True)
    )
    M3 += 0.5 * (
            np.einsum("abe,ec->abc", x2_vvov[:, :, i, :], t2[:, :, j, k], optimize=True)
            -np.einsum("abe,ec->abc", x2_vvov[:, :, j, :], t2[:, :, i, k], optimize=True)
            -np.einsum("abe,ec->abc", x2_vvov[:, :, k, :], t2[:, :, j, i], optimize=True)
    )
    M3 -= 0.5 * (
            np.einsum("am,bcm->abc", x2_vooo[:, :, i, j], t2[:, :, :, k], optimize=True)
            -np.einsum("am,bcm->abc", x2_vooo[:, :, k, j], t2[:, :, :, i], optimize=True)
            -np.einsum("am,bcm->abc", x2_vooo[:, :, i, k], t2[:, :, :, j], optimize=True)
    )
    # antisymmetrize A(abc)
    M3 -= np.transpose(M3, (1, 0, 2)) + np.transpose(M3, (2, 1, 0)) # (a/bc)
    M3 -= np.transpose(M3, (0, 2, 1)) # (bc)
    return M3

def leftamps_ijk(i, j, k, h1_ov, h2_oovv, h2_vovv, h2_ooov, l1, l2):
    '''Computes the left vector amplitudes L3(abc) = <0|(1+L1+L2)H(2)|ijkabc> for a fixed i,j,k for i<j<k.'''
    L3 = 0.5 * (
            np.einsum("eba,ec->abc", h2_vovv[:, i, :, :], l2[:, :, j, k], optimize=True)
            -np.einsum("eba,ec->abc", h2_vovv[:, j, :, :], l2[:, :, i, k], optimize=True)
            -np.einsum("eba,ec->abc", h2_vovv[:, k, :, :], l2[:, :, j, i], optimize=True)
    )
    L3 -= 0.5 * (
            np.einsum("ma,bcm->abc", h2_ooov[j, i, :, :], l2[:, :, :, k], optimize=True)
            -np.einsum("ma,bcm->abc", h2_ooov[k, i, :, :], l2[:, :, :, j], optimize=True)
            -np.einsum("ma,bcm->abc", h2_ooov[j, k, :, :], l2[:, :, :, i], optimize=True)
    )
    L3 += 0.5 * (
            np.einsum("ab,c->abc", h2_oovv[i, j, :, :], l1[:, k], optimize=True)
            -np.einsum("ab,c->abc", h2_oovv[k, j, :, :], l1[:, i], optimize=True)
            -np.einsum("ab,c->abc", h2_oovv[i, k, :, :], l1[:, j], optimize=True)
    )
    L3 += 0.5 * (
            np.einsum("a,bc->abc", h1_ov[i, :], l2[:, :, j, k], optimize=True)
            -np.einsum("a,bc->abc", h1_ov[j, :], l2[:, :, i, k], optimize=True)
            -np.einsum("a,bc->abc", h1_ov[k, :], l2[:, :, j, i], optimize=True)
    )
    # antisymmetrize A(abc)
    L3 -= np.transpose(L3, (1, 0, 2)) + np.transpose(L3, (2, 1, 0)) # (a/bc)
    L3 -= np.transpose(L3, (0, 2, 1)) # (bc)
    return L3

def onebody_denom_abc(fock, v):
    eps = np.diagonal(fock)
    n = np.newaxis
    e_abc = -eps[v, n, n] - eps[n, v, n] - eps[n, n, v]
    return e_abc

def vvvv_denom_abc(h_vvvv):
    nu, _, _, _ = h_vvvv.shape
    eps = np.zeros((nu, nu))
    n = np.newaxis
    # extract this kind of diagonal from h_vvvv
    for a in range(nu):
        for b in range(nu):
            eps[a, b] = h_vvvv[b, a, b, a]
    # form the denominator
    e_abc = -eps[:, :, n] - eps[:, n, :] - eps[n, :, :]
    return e_abc

def voov_denom_abc(i, j, k, h_voov):
    n = np.newaxis
    eps_i = np.diagonal(h_voov[:, i, i, :])
    eps_j = np.diagonal(h_voov[:, j, j, :])
    eps_k = np.diagonal(h_voov[:, k, k, :])
    e_abc = (
            -eps_i[:, n, n] - eps_i[n, :, n] - eps_i[n, n, :]
            -eps_j[:, n, n] - eps_j[n, :, n] - eps_j[n, n, :]
            -eps_k[:, n, n] - eps_k[n, :, n] - eps_k[n, n, :]
    )
    return e_abc

def voo_denom_abc(i, j, k, d3o):
    n = np.newaxis
    eps_ij = d3o[:, i, j]
    eps_ik = d3o[:, i, k]
    eps_jk = d3o[:, j, k]
    e_abc = (
            +eps_ij[:, n, n] + eps_ik[:, n, n] + eps_jk[:, n, n]
            +eps_ij[n, :, n] + eps_ik[n, :, n] + eps_jk[n, :, n]
            +eps_ij[n, n, :] + eps_ik[n, n, :] + eps_jk[n, n, :]
    )
    return e_abc

def vov_denom_abc(i, j, k, d3v):
    nu, no, _ = d3v.shape
    n = np.newaxis
    eps_i = d3v[:, i, :]
    eps_j = d3v[:, j, :]
    eps_k = d3v[:, k, :]
    e_abc = (
            -eps_i[:, :, n] - eps_i[:, n, :] - eps_i[n, :, :]
            -eps_j[:, :, n] - eps_j[:, n, :] - eps_j[n, :, :]
            -eps_k[:, :, n] - eps_k[:, n, :] - eps_k[n, :, :]
    )
    return e_abc

def correction_in_loop(t1, t2, l1, l2, r1, r2, r0, omega, no, nu, fock, H1, H2, I_vooo, X_vooo, X_vvov, d3o, d3v, o, v):

    # precompute blocks of diagonal that do not depend on occupied indices
    denom_A_v = onebody_denom_abc(fock, v)
    denom_B_v = onebody_denom_abc(H1, v)
    denom_C_vvvv = vvvv_denom_abc(H2[v, v, v, v])

    # Compute triples correction in loop
    delta_A = 0.0
    delta_B = 0.0
    delta_C = 0.0
    delta_D = 0.0
    error = 0.0
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):

                denom_A_o = fock[o, o][i, i] + fock[o, o][j, j] + fock[o, o][k, k] 
                denom_B_o = H1[o, o][i, i] + H1[o, o][j, j] + H1[o, o][k, k]
                denom_C_voov = voov_denom_abc(i, j, k, H2[v, o, o, v])
                denom_C_oooo = -H2[o, o, o, o][j, i, j, i] - H2[o, o, o, o][k, i, k, i] - H2[o, o, o, o][k, j, k, j]
                denom_D_voo = voo_denom_abc(i, j, k, d3o)
                denom_D_vov = vov_denom_abc(i, j, k, d3v)

                m3 = moments_ijk(i, j, k, I_vooo, H2[v, o, o, o], H2[v, v, o, v], X_vooo, X_vvov, t2, r2, r0)
                l3 = leftamps_ijk(i, j, k, H1[o, v], H2[o, o, v, v], H2[v, o, v, v], H2[o, o, o, v], l1, l2)
                LM = m3 * l3

                delta_A += (1.0 / 6.0) * np.sum(LM/(omega + denom_A_o + denom_A_v))
                delta_B += (1.0 / 6.0) * np.sum(LM/(omega + denom_B_o + denom_B_v))
                delta_C += (1.0 / 6.0) * np.sum(LM/(omega + denom_B_o + denom_B_v + denom_C_voov + denom_C_oooo + denom_C_vvvv))
                delta_D += (1.0 / 6.0) * np.sum(LM/(omega + denom_B_o + denom_B_v + denom_C_voov + denom_C_oooo + denom_C_vvvv + denom_D_voo + denom_D_vov))

    return delta_A, delta_B, delta_C, delta_D
