import numpy as np
from miniccpy.hbar_diagonal import (get_3body_hbar_triples_diagonal,
                                    vv_denom_abc,
                                    vvvv_denom_abc,
                                    voov_denom_abc_3p2h,
                                    voo_denom_abc_3p2h,
                                    vov_denom_abc_3p2h)

def kernel(T, R, L, r0, omega, fock, H1, H2, o, v):

    t1, t2 = T
    r1, r2 = R
    l1, l2 = L
    # orbital dimensions
    nu, no = t1.shape
    # get 3-body Hbar triples diagonal
    d3v, d3o = get_3body_hbar_triples_diagonal(H2[o, o, v, v], t2)

    # Moment
    I2_voo = (
                -np.einsum("amje,e->amj", H2[v, o, o, v], r1, optimize=True) # (!)
                + 0.5 * np.einsum("amef,efj->amj", H2[v, o, v, v], r2, optimize=True)
                + np.einsum("mnje,aen->amj", H2[o, o, o, v], r2, optimize=True)
    )
    I2_vvv = (
                0.5 * np.einsum("bcef,e->bcf", H2[v, v, v, v], r1, optimize=True)
                - np.einsum("cmef,bem->bcf", H2[v, o, v, v], r2, optimize=True)
    )
    I2_vvv -= np.transpose(I2_vvv, (1, 0, 2))

    # M3 = -(3.0 / 12.0) * np.einsum("cmkj,abm->abcjk", H2[v, o, o, o], r2, optimize=True)     # (7)
    # M3 += (6.0 / 12.0) * np.einsum("cbke,aej->abcjk", H2[v, v, o, v], r2, optimize=True)     # (8)
    # M3 -= (6.0 / 12.0) * np.einsum("amj,bcmk->abcjk", I2_voo, t2, optimize=True) # (9)
    # M3 += (3.0 / 12.0) * np.einsum("abe,ecjk->abcjk", I2_vvv, t2, optimize=True) # (10)
    # M3 -= np.transpose(M3, (1, 0, 2, 3, 4)) + np.transpose(M3, (2, 1, 0, 3, 4)) # antisymmetrize A(a/bc)
    # M3 -= np.transpose(M3, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    # M3 -= np.transpose(M3, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
   
    # Left Vector
    # L3 = (3.0 / 12.0) * np.einsum("a,jkbc->abcjk", l1, H2[o, o, v, v], optimize=True)
    # L3 += (6.0 / 12.0) * np.einsum("abj,kc->abcjk", l2, H1[o, v], optimize=True)
    # L3 -= (3.0 / 12.0) * np.einsum("abm,jkmc->abcjk", l2, H2[o, o, o, v], optimize=True)
    # L3 += (6.0 / 12.0) * np.einsum("eck,ejab->abcjk", l2, H2[v, o, v, v], optimize=True)
    # L3 -= np.transpose(L3, (1, 0, 2, 3, 4)) + np.transpose(L3, (2, 1, 0, 3, 4)) # antisymmetrize A(a/bc)
    # L3 -= np.transpose(L3, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    # L3 -= np.transpose(L3, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)

    delta_A, delta_B, delta_C, delta_D = correction_in_loop(l1, l2, t2, r2, omega, no, nu, fock, H1, H2, I2_voo, I2_vvv, d3o, d3v, o, v)

    # Store triples corrections in dictionary
    delta_T = {"A": delta_A, "B": delta_B, "C": delta_C, "D": delta_D}
    return delta_T

def moments_jk(j, k, h2_vooo, h2_vvov, I_voo, I_vvv, t2, r2):
    '''Computes the moment M(abc) = <ijkabc|H(2)|0> for a fixed i,j,k for i<j<k.'''
    M3 = -0.5 * (
        np.einsum("cm,abm->abc", h2_vooo[:, :, k, j], r2, optimize=True)
    )
    M3 += 0.5 * (
        np.einsum("cbe,ae->abc", h2_vvov[:, :, k, :], r2[:, :, j], optimize=True)
        - np.einsum("cbe,ae->abc", h2_vvov[:, :, j, :], r2[:, :, k], optimize=True)
    )
    M3 -= 0.5 * (
        np.einsum("am,bcm->abc", I_voo[:, :, j], t2[:, :, :, k], optimize=True)
        - np.einsum("am,bcm->abc", I_voo[:, :, k], t2[:, :, :, j], optimize=True)
    )
    M3 += 0.5 * (
        np.einsum("abe,ec->abc", I_vvv, t2[:, :, j, k], optimize=True)
    )
    # antisymmetrize A(abc)
    M3 -= np.transpose(M3, (1, 0, 2)) + np.transpose(M3, (2, 1, 0)) # (a/bc)
    M3 -= np.transpose(M3, (0, 2, 1)) # (bc)
    return M3

def leftamps_jk(j, k, h1_ov, h2_ooov, h2_vovv, h2_oovv, l1, l2):
    '''Computes the moment M(abc) = <ijkabc|H(2)|0> for a fixed i,j,k for i<j<k.'''
    L3 = 0.5 * (
        np.einsum("a,bc->abc", l1, h2_oovv[j, k, :, :], optimize=True)
    )
    L3 += 0.5 * (
        np.einsum("ab,c->abc", l2[:, :, j], h1_ov[k, :], optimize=True)
        - np.einsum("ab,c->abc", l2[:, :, k], h1_ov[j, :], optimize=True)
    )
    L3 -= 0.5 * (
        np.einsum("abm,mc->abc", l2, h2_ooov[j, k, :, :], optimize=True)
    )
    L3 += 0.5 * (
        np.einsum("ec,eab->abc", l2[:, :, k], h2_vovv[:, j, :, :], optimize=True)
        - np.einsum("ec,eab->abc", l2[:, :, j], h2_vovv[:, k, :, :], optimize=True)
    )
    # antisymmetrize A(abc)
    L3 -= np.transpose(L3, (1, 0, 2)) + np.transpose(L3, (2, 1, 0)) # (a/bc)
    L3 -= np.transpose(L3, (0, 2, 1)) # (bc)
    return L3

def correction_in_loop(l1, l2, t2, r2, omega, no, nu, fock, H1, H2, I_voo, I_vvv, d3o, d3v, o, v):

    # precompute blocks of diagonal that do not depend on occupied indices
    denom_A_v = vv_denom_abc(fock, v)
    denom_B_v = vv_denom_abc(H1, v)
    denom_C_vvvv = vvvv_denom_abc(H2[v, v, v, v])

    # Compute triples correction in loop
    delta_A = 0.0
    delta_B = 0.0
    delta_C = 0.0
    delta_D = 0.0
    for j in range(no):
        for k in range(j + 1, no):

            denom_A_o = fock[o, o][j, j] + fock[o, o][k, k]
            denom_B_o = H1[o, o][j, j] + H1[o, o][k, k]
            denom_C_voov = voov_denom_abc_3p2h(j, k, H2[v, o, o, v])
            denom_C_oooo = -H2[o, o, o, o][k, j, k, j]
            denom_D_voo = voo_denom_abc_3p2h(j, k, d3o)
            denom_D_vov = vov_denom_abc_3p2h(j, k, d3v)

            m3 = moments_jk(j, k, H2[v, o, o, o], H2[v, v, o, v], I_voo, I_vvv, t2, r2)
            l3 = leftamps_jk(j, k, H1[o, v], H2[o, o, o, v], H2[v, o, v, v], H2[o, o, v, v], l1, l2)
            LM = m3 * l3

            delta_A += (1.0 / 6.0) * np.sum(LM/(omega + denom_A_o + denom_A_v))
            delta_B += (1.0 / 6.0) * np.sum(LM/(omega + denom_B_o + denom_B_v))
            delta_C += (1.0 / 6.0) * np.sum(LM/(omega + denom_B_o + denom_B_v + denom_C_voov + denom_C_oooo + denom_C_vvvv))
            delta_D += (1.0 / 6.0) * np.sum(LM/(omega + denom_B_o + denom_B_v + denom_C_voov + denom_C_oooo + denom_C_vvvv + denom_D_voo + denom_D_vov))

            # Equivalent form of D denominator
            # for a in range(nu):
            #     for b in range(a + 1, nu):
            #         for c in range(b + 1, nu):
            #             denom_A = denom_A_o + denom_A_v[a, b, c]
            #             denom_B = denom_B_o + denom_B_v[a, b, c]
            #             denom_C = denom_B + denom_C_voov[a, b, c] + denom_C_oooo + denom_C_vvvv[a, b, c]
            #             denom_D = denom_C + (
            #                 d3o[a, j, k] + d3o[b, j, k] + d3o[c, j, k]
            #                 -d3v[a, j, b] - d3v[a, j, c] - d3v[b, j, c]
            #                 -d3v[a, k, b] - d3v[a, k, c] - d3v[b, k, c]
            #             )
            #             delta_D += LM[a, b, c]/(omega + denom_D)

    return delta_A, delta_B, delta_C, delta_D
