import numpy as np
from miniccpy.hbar_diagonal import get_3body_hbar_triples_diagonal, vv_denom_abc, vvvv_denom_abc, voov_denom_abc, voo_denom_abc, vov_denom_abc

def kernel(T, L, fock, H1, H2, o, v, nacto, nactu, num_active):

    # unpack T and L vectors
    t1, t2 = T
    l1, l2 = L

    # orbial dimensions
    nu, no = t1.shape

    # get 3-body Hbar triples diagonal
    d3v, d3o = get_3body_hbar_triples_diagonal(H2[o, o, v, v], t2)

    # Compute adjusted h(vooo) intermediate
    I_vooo = H2[v, o, o, o] - np.einsum("me,aeij->amij", H1[o, v], t2, optimize=True)

    # Perform correction in loop
    delta_A, delta_B, delta_C, delta_D = correction_in_loop(t1, t2, l1, l2, fock, H1, I_vooo, H2, d3o, d3v, o, v, no, nu, nacto, nactu, num_active)

    # Store triples corrections in dictionary
    delta_T = {"A": delta_A, "B": delta_B, "C": delta_C, "D": delta_D}
    return delta_T

def is_active_occ(i, no, nacto):
    if i < no - nacto:
        return 0
    else:
        return 1

def is_active_unocc(a, nu, nactu):
    if a < nactu:
        return 1
    else:
        return 0

def abc_mask(nu, nactu, num_active):
    mask = np.zeros((nu, nu, nu))
    for a in range(nu):
        for b in range(a + 1, nu):
            for c in range(b + 1, nu):

                ia = is_active_unocc(a, nu, nactu)
                ib = is_active_unocc(b, nu, nactu)
                ic = is_active_unocc(c, nu, nactu)
                n = ia + ib + ic

                if n < num_active:
                    mask[a, b, c] = 1.0
                    mask[a, c, b] = 1.0
                    mask[b, a, c] = 1.0
                    mask[b, c, a] = 1.0
                    mask[c, b, a] = 1.0
                    mask[c, a, b] = 1.0
    return mask
    

def moments_ijk(i, j, k, I_vooo, h2_vvov, t2):
    '''Computes the moment M(abc) = <ijkabc|H(2)|0> for a fixed i,j,k for i<j<k.'''
    M3 = 0.5 * (
            np.einsum("abe,ec->abc", h2_vvov[:, :, i, :], t2[:, :, j, k], optimize=True)
            -np.einsum("abe,ec->abc", h2_vvov[:, :, j, :], t2[:, :, i, k], optimize=True)
            -np.einsum("abe,ec->abc", h2_vvov[:, :, k, :], t2[:, :, j, i], optimize=True)
    )
    M3 -= 0.5 * (
            np.einsum("am,bcm->abc", I_vooo[:, :, i, j], t2[:, :, :, k], optimize=True)
            -np.einsum("am,bcm->abc", I_vooo[:, :, k, j], t2[:, :, :, i], optimize=True)
            -np.einsum("am,bcm->abc", I_vooo[:, :, i, k], t2[:, :, :, j], optimize=True)
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

def correction_in_loop(t1, t2, l1, l2, fock, H1, I_vooo, H2, d3o, d3v, o, v, no, nu, nacto, nactu, num_active):

    # precompute blocks of diagonal that do not depend on occupied indices
    denom_A_v = vv_denom_abc(fock, v)
    denom_B_v = vv_denom_abc(H1, v)
    denom_C_vvvv = vvvv_denom_abc(H2[v, v, v, v])

    # Compute triples correction in loop
    delta_A = 0.0
    delta_B = 0.0
    delta_C = 0.0
    delta_D = 0.0
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):

                # create abc mask
                mask = np.ones((nu, nu, nu))

                # check if (i,j,k) has at least one active index
                n_occ = (is_active_occ(i, no, nacto)
                         + is_active_occ(j, no, nacto)
                         + is_active_occ(k, no, nacto))
    
                # compute i,j,k part of triples denominator
                denom_A_o = fock[o, o][i, i] + fock[o, o][j, j] + fock[o, o][k, k] 
                denom_B_o = H1[o, o][i, i] + H1[o, o][j, j] + H1[o, o][k, k]
                denom_C_voov = voov_denom_abc(i, j, k, H2[v, o, o, v])
                denom_C_oooo = -H2[o, o, o, o][j, i, j, i] - H2[o, o, o, o][k, i, k, i] - H2[o, o, o, o][k, j, k, j]
                denom_D_voo = voo_denom_abc(i, j, k, d3o)
                denom_D_vov = vov_denom_abc(i, j, k, d3v)

                # compute a,b,c part of moments and left vector
                m3 = moments_ijk(i, j, k, I_vooo, H2[v, v, o, v], t2)
                l3 = leftamps_ijk(i, j, k, H1[o, v], H2[o, o, v, v], H2[v, o, v, v], H2[o, o, o, v], l1, l2)
                LM = m3 * l3

                # set mask to zero out |ijKAbc> contributions
                for a in range(nu):
                    for b in range(a + 1, nu):
                        for c in range(b + 1, nu):
                            n_unocc = (is_active_unocc(a, nu, nactu)
                                     + is_active_unocc(b, nu, nactu)
                                     + is_active_unocc(c, nu, nactu))
                            if n_occ >= num_active and n_unocc >= num_active:
                                mask[a, b, c] = 0.0
                                mask[a, c, b] = 0.0
                                mask[b, c, a] = 0.0
                                mask[b, a, c] = 0.0
                                mask[c, a, b] = 0.0
                                mask[c, b, a] = 0.0
                # apply mask
                LM *= mask

                # compute corrections in a vectorized manner
                delta_A += (1.0 / 6.0) * np.sum(LM/(denom_A_o + denom_A_v))
                delta_B += (1.0 / 6.0) * np.sum(LM/(denom_B_o + denom_B_v))
                delta_C += (1.0 / 6.0) * np.sum(LM/(denom_B_o + denom_B_v + denom_C_voov + denom_C_oooo + denom_C_vvvv))
                delta_D += (1.0 / 6.0) * np.sum(LM/(denom_B_o + denom_B_v + denom_C_voov + denom_C_oooo + denom_C_vvvv + denom_D_voo + denom_D_vov))

    return delta_A, delta_B, delta_C, delta_D
