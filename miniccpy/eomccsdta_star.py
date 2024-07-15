import numpy as np
from miniccpy.hbar_diagonal import vv_denom_abc

def kernel(T, R, L, r0, omega, fock, g, H1, H2, o, v):
    '''
    Performs the EE-EOMCCSD(T)a* noniterative triples correction to the EOMCCSD energetics based
    on using the CCSD(T)(a) similarity-transformed Hamiltonian. The original paper for this method
    is Matthews and Stanton, J. Chem. Phys. 145, 124102 (2016). The manuscript Zhao and Matthews,
    "Analytic gradients for equation-of-motion coupled cluster with single, double and perturbative
    triple excitations", arXiV preprint [https://doi.org/10.48550/arXiv.2406.05595] provides the
    EOMCCSD* equations directly in Eqs. (11)-(13). Here, I am assuming that the EOMCCSD(T)a* formulas
    are analogous to EOMCCSD*, except that the CCSD(T)(a) HBar is used instead of CCSD.
    L3 = < 0 | (V_N*L1)_DC + (V_N*L2)_C | ijkabc >
    M3 = < ijkabc | (V_N*R2)_C + (V_N*T2*R1)_C | 0 >
    delta = < 0 | L3*M3 | 0 >`
    '''

    t1, t2 = T
    r1, r2 = R
    l1, l2 = L
    # orbital dimensions
    nu, no = t1.shape
    # # MP denominator
    # eps = np.diagonal(fock)
    # n = np.newaxis
    # e_abcijk = (eps[v, n, n, n, n, n] + eps[n, v, n, n, n, n] + eps[n, n, v, n, n, n]
    #            - eps[n, n, n, o, n, n] - eps[n, n, n, n, o, n] - eps[n, n, n, n, n, o])

    # Intermediates
    # X_ov = np.einsum("mnef,fn->me", H2[o, o, v, v], r1, optimize=True) # 1st order

    X_vvov =(
        np.einsum("amje,bm->baje", g[v, o, o, v], r1, optimize=True) # 2+0 = 2nd order
        # + np.einsum("amfe,bejm->bajf", g[v, o, v, v], r2, optimize=True) # 2+1 = 3rd order
        + 0.5 * np.einsum("abfe,ej->bajf", g[v, v, v, v], r1, optimize=True)
        # + 0.25 * np.einsum("nmje,abmn->baje", g[o, o, o, v], r2, optimize=True)
        # - 0.5 * np.einsum("me,abmj->baje", X_ov, t2, optimize=True)
    )
    X_vvov -= np.transpose(X_vvov, (1, 0, 2, 3))

    X_vooo = (
        -np.einsum("bmie,ej->bmji", g[v, o, o, v], r1, optimize=True)
        # +np.einsum("nmie,bejm->bnji", g[o, o, o, v], r2, optimize=True)
        - 0.5 * np.einsum("nmij,bm->bnji", g[o, o, o, o], r1, optimize=True)
        # + 0.25 * np.einsum("bmfe,efij->bmji", g[v, o, v, v], r2, optimize=True)
    )
    X_vooo -= np.transpose(X_vooo, (0, 1, 3, 2))

    # # Excited-state moment
    # M3 = 0.25 * np.einsum("baje,ecik->abcijk", X_vvov, t2, optimize=True)
    # M3 += 0.25 * np.einsum("baje,ecik->abcijk", g[v, v, o, v], r2, optimize=True)
    # M3 -= 0.25 * np.einsum("bmji,acmk->abcijk", X_vooo, t2, optimize=True)
    # M3 -= 0.25 * np.einsum("bmji,acmk->abcijk", g[v, o, o, o], r2, optimize=True)
    # M3 -= np.transpose(M3, (0, 1, 2, 3, 5, 4)) # (jk)
    # M3 -= np.transpose(M3, (0, 1, 2, 4, 3, 5)) + np.transpose(M3, (0, 1, 2, 5, 4, 3)) # (i/jk)
    # M3 -= np.transpose(M3, (0, 2, 1, 3, 4, 5)) # (bc)
    # M3 -= np.transpose(M3, (2, 1, 0, 3, 4, 5)) + np.transpose(M3, (1, 0, 2, 3, 4, 5)) # (a/bc)
    #
    # # Left
    # L3 = (9.0 / 36.0) * (
    #       np.einsum("ijab,ck->abcijk", g[o, o, v, v], l1, optimize=True)
    #       # +np.einsum("ia,bcjk->abcijk", I_ov, l2, optimize=True)
    #       +np.einsum("eiba,ecjk->abcijk", g[v, o, v, v], l2, optimize=True)
    #       -np.einsum("jima,bcmk->abcijk", g[o, o, o, v], l2, optimize=True)
    # )
    # L3 -= np.transpose(L3, (0, 1, 2, 3, 5, 4)) # (jk)
    # L3 -= np.transpose(L3, (0, 1, 2, 4, 3, 5)) + np.transpose(L3, (0, 1, 2, 5, 4, 3)) # (i/jk)
    # L3 -= np.transpose(L3, (0, 2, 1, 3, 4, 5)) # (bc)
    # L3 -= np.transpose(L3, (2, 1, 0, 3, 4, 5)) + np.transpose(L3, (1, 0, 2, 3, 4, 5)) # (a/bc)
    #
    # # divide L3 by MP denominator
    # L3 /= (omega - e_abcijk)

    # contract L3*M3
    # delta_A = (1.0 / 36.0) * np.einsum("abcijk,abcijk->", L3, M3, optimize=True)
    delta_A = correction_in_loop(t1, t2, l1, l2, r1, r2, omega, no, nu, fock, g, X_vooo, X_vvov, o, v)

    # Store triples corrections in dictionary
    delta_T = {"A": delta_A, "B": 0.0, "C": 0.0, "D": 0.0}
    return delta_T


def moments_ijk(i, j, k, h2_vooo, h2_vvov, x2_vooo, x2_vvov, t2, r2):
    '''Computes the moment M(abc) = <ijkabc|H(2)|0> for a fixed i,j,k for i<j<k.'''
    M3 = 0.5 * (
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

def leftamps_ijk(i, j, k, h2_oovv, h2_vovv, h2_ooov, l1, l2):
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
    # L3 += 0.5 * (
    #         np.einsum("a,bc->abc", h1_ov[i, :], l2[:, :, j, k], optimize=True)
    #         -np.einsum("a,bc->abc", h1_ov[j, :], l2[:, :, i, k], optimize=True)
    #         -np.einsum("a,bc->abc", h1_ov[k, :], l2[:, :, j, i], optimize=True)
    # )
    # antisymmetrize A(abc)
    L3 -= np.transpose(L3, (1, 0, 2)) + np.transpose(L3, (2, 1, 0)) # (a/bc)
    L3 -= np.transpose(L3, (0, 2, 1)) # (bc)
    return L3

def correction_in_loop(t1, t2, l1, l2, r1, r2, omega, no, nu, fock, g, X_vooo, X_vvov, o, v):

    # precompute blocks of diagonal that do not depend on occupied indices
    denom_A_v = vv_denom_abc(fock, v)

    # Compute triples correction in loop
    delta_A = 0.0

    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):

                denom_A_o = fock[o, o][i, i] + fock[o, o][j, j] + fock[o, o][k, k]

                m3 = moments_ijk(i, j, k, g[v, o, o, o], g[v, v, o, v], X_vooo, X_vvov, t2, r2)
                l3 = leftamps_ijk(i, j, k, g[o, o, v, v], g[v, o, v, v], g[o, o, o, v], l1, l2)
                LM = m3 * l3

                delta_A += (1.0 / 6.0) * np.sum(LM/(omega + denom_A_o + denom_A_v))

    return delta_A
