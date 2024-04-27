import numpy as np
from miniccpy.hbar_diagonal import vv_denom_abc

def kernel(T, L, fock, H1, g, o, v):
    # Note: H1 should just be None. It's not even used. It's just there
    # to make the call in run_correction the same for CCSD(T) as for CR-CC(2,3).

    # unpack T amplitudes
    t1, t2 = T
    # Perform correction in loop
    delta_A = correction_in_loop(t1, t2, fock, g, o, v)
    # Store triples corrections in dictionary
    delta_T = {"A": delta_A, "B": 0.0, "C": 0.0, "D": 0.0}
    return delta_T

def moments_ijk(i, j, k, g_vooo, g_vvov, t2):
    '''Computes the leading part of the moment M(abc) = <ijkabc|(V*T2)_C|0> for a fixed i,j,k for i<j<k.'''
    M3 = 0.5 * (
            np.einsum("abe,ec->abc", g_vvov[:, :, i, :], t2[:, :, j, k], optimize=True)
            -np.einsum("abe,ec->abc", g_vvov[:, :, j, :], t2[:, :, i, k], optimize=True)
            -np.einsum("abe,ec->abc", g_vvov[:, :, k, :], t2[:, :, j, i], optimize=True)
    )
    M3 -= 0.5 * (
            np.einsum("am,bcm->abc", g_vooo[:, :, i, j], t2[:, :, :, k], optimize=True)
            -np.einsum("am,bcm->abc", g_vooo[:, :, k, j], t2[:, :, :, i], optimize=True)
            -np.einsum("am,bcm->abc", g_vooo[:, :, i, k], t2[:, :, :, j], optimize=True)
    )
    # antisymmetrize A(abc)
    M3 -= np.transpose(M3, (1, 0, 2)) + np.transpose(M3, (2, 1, 0)) # (a/bc)
    M3 -= np.transpose(M3, (0, 2, 1)) # (bc)
    return M3

def leftamps_dc_ijk(i, j, k, f_ov, g_oovv, l1, l2):
    '''Computes the disconnected left vector amplitudes L3(abc) = <0|(1+L1+L2)H(2)|ijkabc> for a fixed i,j,k for i<j<k.'''
    L3 = 0.5 * (
            np.einsum("ab,c->abc", g_oovv[i, j, :, :], l1[:, k], optimize=True)
            -np.einsum("ab,c->abc", g_oovv[k, j, :, :], l1[:, i], optimize=True)
            -np.einsum("ab,c->abc", g_oovv[i, k, :, :], l1[:, j], optimize=True)
    )
    # Note: This term is usually ignored in most CCSD(T) formulas because they 
    #       assume Brillouin's theorem holds. However, for non-Brillouin orbitals,
    #       (e.g., ROHF), these need to be included to get the right answer, as 
    #       these enter in 4th-order MBPT.
    L3 += 0.5 * (
            np.einsum("a,bc->abc", f_ov[i, :], l2[:, :, j, k], optimize=True)
            -np.einsum("a,bc->abc", f_ov[j, :], l2[:, :, i, k], optimize=True)
            -np.einsum("a,bc->abc", f_ov[k, :], l2[:, :, j, i], optimize=True)
    )
    # antisymmetrize A(abc)
    L3 -= np.transpose(L3, (1, 0, 2)) + np.transpose(L3, (2, 1, 0)) # (a/bc)
    L3 -= np.transpose(L3, (0, 2, 1)) # (bc)
    return L3

def correction_in_loop(t1, t2, fock, g, o, v):
    # orbital dimensions
    no, nu = fock[o, v].shape
    # precompute blocks of diagonal that do not depend on occupied indices
    denom_A_v = vv_denom_abc(fock, v)
    # Compute triples correction in loop
    delta_A = 0.0
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                # compute i,j,k part of triples denominator
                denom_A_o = fock[o, o][i, i] + fock[o, o][j, j] + fock[o, o][k, k] 
                # compute a,b,c part of moments and left vector
                m3 = moments_ijk(i, j, k, g[v, o, o, o], g[v, v, o, v], t2)
                l3_dc = leftamps_dc_ijk(i, j, k, fock[o, v], g[o, o, v, v], t1, t2)
                # the connected part of l3 is equal to m3.conj() 
                LM = m3 * (m3 + l3_dc) 
                # compute corrections in a vectorized manner
                delta_A += (1.0 / 6.0) * np.sum(LM/(denom_A_o + denom_A_v))

    return delta_A
