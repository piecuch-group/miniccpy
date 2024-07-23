import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def add_t3_contributions(np.ndarray[np.float64_t, ndim=2] t1,
                         np.ndarray[np.float64_t, ndim=4] t2,
                         np.ndarray[np.float64_t, ndim=2] f,
                         np.ndarray[np.float64_t, ndim=4] g,
                         np.ndarray[np.float64_t, ndim=4] I_vooo,
                         np.ndarray[np.float64_t, ndim=4] I_vvov,
                         np.ndarray[np.float64_t, ndim=3] e_abc,
                         o, v):
    # variable definitions
    cdef int i, j, k, a, no, nu
    cdef double denom_occ
    cdef np.ndarray[np.float64_t, ndim=2] h_ov
    cdef np.ndarray[np.float64_t, ndim=4] h_ooov
    cdef np.ndarray[np.float64_t, ndim=4] h_vovv
    cdef np.ndarray[np.float64_t, ndim=4] gs_oovv
    cdef np.ndarray[np.float64_t, ndim=3] m3
    cdef np.ndarray[np.float64_t, ndim=2] singles_res
    cdef np.ndarray[np.float64_t, ndim=4] doubles_res

    nu = t1.shape[0]
    no = t1.shape[1]
    # Intermediates
    h_ov = f[o, v] + (
              2.0 * np.einsum("mnef,fn->me", g[o, o, v, v], t1, optimize=True)
            - np.einsum("mnfe,fn->me", g[o, o, v, v], t1, optimize=True)
    )
    h_ooov = g[o, o, o, v] + np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    h_vovv = g[v, o, v, v] - np.einsum("nmef,an->amef", g[o, o, v, v], t1, optimize=True)
    # RHF-adapted integral element
    gs_oovv = 2.0 * g[o, o, v, v] - g[o, o, v, v].swapaxes(2, 3)
    # residual containers
    singles_res = np.zeros((nu, no))
    doubles_res = np.zeros((nu, nu, no, no))
    for i in range(no):
        for j in range(no):
            for k in range(no):
                if i == j and j == k: continue
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                # -h2(amij) * t2(bcmk)
                m3 = -np.einsum("am,bcm->abc", I_vooo[:, :, i, j], t2[:, :, :, k], optimize=True) # (1)
                m3 -= np.einsum("bm,acm->abc", I_vooo[:, :, j, i], t2[:, :, :, k], optimize=True) # (ij)(ab)
                m3 -= np.einsum("cm,bam->abc", I_vooo[:, :, k, j], t2[:, :, :, i], optimize=True) # (ac)(ik)
                m3 -= np.einsum("am,cbm->abc", I_vooo[:, :, i, k], t2[:, :, :, j], optimize=True) # (bc)(jk)
                m3 -= np.einsum("bm,cam->abc", I_vooo[:, :, j, k], t2[:, :, :, i], optimize=True) # (ij)(ab)(ac)(ik)
                m3 -= np.einsum("cm,abm->abc", I_vooo[:, :, k, i], t2[:, :, :, j], optimize=True) # (ij)(ab)(bc)(jk)
                # h2(abie) * t2(bcek)
                m3 += np.einsum("abe,ec->abc", I_vvov[:, :, i, :], t2[:, :, j, k], optimize=True) # (1)
                m3 += np.einsum("bae,ec->abc", I_vvov[:, :, j, :], t2[:, :, i, k], optimize=True) # (ij)(ab)
                m3 += np.einsum("cbe,ea->abc", I_vvov[:, :, k, :], t2[:, :, j, i], optimize=True) # (ac)(ik)
                m3 += np.einsum("ace,eb->abc", I_vvov[:, :, i, :], t2[:, :, k, j], optimize=True) # (bc)(jk)
                m3 += np.einsum("bce,ea->abc", I_vvov[:, :, j, :], t2[:, :, k, i], optimize=True) # (ij)(ab)(ac)(ik)
                m3 += np.einsum("cae,eb->abc", I_vvov[:, :, k, :], t2[:, :, i, j], optimize=True) # (ij)(ab)(bc)(jk)
                # divide by MP denominator
                m3 /= (e_abc + denom_occ)
                # zero out diagonal elements
                for a in range(nu):
                    m3[a, a, a] *= 0.0
                # update singles residual
                singles_res[:, i] += np.einsum('abc,bc->a', m3 - m3.swapaxes(0, 2), gs_oovv[j, k, :, :], optimize=True)
                # symmetrize
                m3 = (2.0 * m3
                      - m3.swapaxes(1, 2)
                      - m3.swapaxes(0, 2)
                )
                # update doubles residual
                doubles_res[:, :, i, j] += 0.5 * np.einsum('abc,c->ab', m3, h_ov[k, :])
                doubles_res[:, :, i, j] += np.einsum('abc,dbc->ad', m3, h_vovv[:, k, :, :])
                doubles_res[:, :, i, :] -= np.einsum('abc,lc->abl', m3, h_ooov[j, k, :, :])
    # Apply (ij)(ab) symmetrizer
    doubles_res += doubles_res.transpose(1, 0, 3, 2)
    return singles_res, doubles_res

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def add_r3_contributions(r1, r2, t1, t2, omega, f, g, H1, H2, h_vvov, h_vooo, x_vvov, x_vooo, e_abc, o, v):
#     nu, no = t1.shape
#     # RHF-adapted integral element
#     gs_oovv = 2.0 * g[o, o, v, v] - g[o, o, v, v].swapaxes(2, 3)
#     # residual containers
#     X1 = np.zeros((nu, no))
#     X2 = np.zeros((nu, nu, no, no))
#     for i in range(no):
#         for j in range(no):
#             for k in range(no):
#                 if i == j and j == k: continue
#                 denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
#                 # -h2(amij) * r2(bcmk)
#                 m3 = -np.einsum("am,bcm->abc", h_vooo[:, :, i, j], r2[:, :, :, k], optimize=True) # (1)
#                 m3 -= np.einsum("bm,acm->abc", h_vooo[:, :, j, i], r2[:, :, :, k], optimize=True) # (ij)(ab)
#                 m3 -= np.einsum("cm,bam->abc", h_vooo[:, :, k, j], r2[:, :, :, i], optimize=True) # (ac)(ik)
#                 m3 -= np.einsum("am,cbm->abc", h_vooo[:, :, i, k], r2[:, :, :, j], optimize=True) # (bc)(jk)
#                 m3 -= np.einsum("bm,cam->abc", h_vooo[:, :, j, k], r2[:, :, :, i], optimize=True) # (ij)(ab)(ac)(ik)
#                 m3 -= np.einsum("cm,abm->abc", h_vooo[:, :, k, i], r2[:, :, :, j], optimize=True) # (ij)(ab)(bc)(jk)
#                 # -x2(amij) * t2(bcmk)
#                 m3 -= np.einsum("am,bcm->abc", x_vooo[:, :, i, j], t2[:, :, :, k], optimize=True) # (1)
#                 m3 -= np.einsum("bm,acm->abc", x_vooo[:, :, j, i], t2[:, :, :, k], optimize=True) # (ij)(ab)
#                 m3 -= np.einsum("cm,bam->abc", x_vooo[:, :, k, j], t2[:, :, :, i], optimize=True) # (ac)(ik)
#                 m3 -= np.einsum("am,cbm->abc", x_vooo[:, :, i, k], t2[:, :, :, j], optimize=True) # (bc)(jk)
#                 m3 -= np.einsum("bm,cam->abc", x_vooo[:, :, j, k], t2[:, :, :, i], optimize=True) # (ij)(ab)(ac)(ik)
#                 m3 -= np.einsum("cm,abm->abc", x_vooo[:, :, k, i], t2[:, :, :, j], optimize=True) # (ij)(ab)(bc)(jk)
#                 # h2(abie) * r2(bcek)
#                 m3 += np.einsum("abe,ec->abc", h_vvov[:, :, i, :], r2[:, :, j, k], optimize=True) # (1)
#                 m3 += np.einsum("bae,ec->abc", h_vvov[:, :, j, :], r2[:, :, i, k], optimize=True) # (ij)(ab)
#                 m3 += np.einsum("cbe,ea->abc", h_vvov[:, :, k, :], r2[:, :, j, i], optimize=True) # (ac)(ik)
#                 m3 += np.einsum("ace,eb->abc", h_vvov[:, :, i, :], r2[:, :, k, j], optimize=True) # (bc)(jk)
#                 m3 += np.einsum("bce,ea->abc", h_vvov[:, :, j, :], r2[:, :, k, i], optimize=True) # (ij)(ab)(ac)(ik)
#                 m3 += np.einsum("cae,eb->abc", h_vvov[:, :, k, :], r2[:, :, i, j], optimize=True) # (ij)(ab)(bc)(jk)
#                 # x2(abie) * t2(bcek)
#                 m3 += np.einsum("abe,ec->abc", x_vvov[:, :, i, :], t2[:, :, j, k], optimize=True) # (1)
#                 m3 += np.einsum("bae,ec->abc", x_vvov[:, :, j, :], t2[:, :, i, k], optimize=True) # (ij)(ab)
#                 m3 += np.einsum("cbe,ea->abc", x_vvov[:, :, k, :], t2[:, :, j, i], optimize=True) # (ac)(ik)
#                 m3 += np.einsum("ace,eb->abc", x_vvov[:, :, i, :], t2[:, :, k, j], optimize=True) # (bc)(jk)
#                 m3 += np.einsum("bce,ea->abc", x_vvov[:, :, j, :], t2[:, :, k, i], optimize=True) # (ij)(ab)(ac)(ik)
#                 m3 += np.einsum("cae,eb->abc", x_vvov[:, :, k, :], t2[:, :, i, j], optimize=True) # (ij)(ab)(bc)(jk)
#                 # divide by MP denominator
#                 m3 /= (omega + e_abc + denom_occ)
#                 # zero out diagonal elements
#                 for a in range(nu):
#                     m3[a, a, a] *= 0.0
#                 # update singles residual
#                 X1[:, i] += np.einsum('abc,bc->a', m3 - m3.swapaxes(0, 2), gs_oovv[j, k, :, :], optimize=True)
#                 # symmetrize
#                 m3 = (2.0 * m3
#                       - m3.swapaxes(1, 2)
#                       - m3.swapaxes(0, 2)
#                 )
#                 # update doubles residual
#                 X2[:, :, i, j] += 0.5 * np.einsum('abc,c->ab', m3, H1[o, v][k, :])
#                 X2[:, :, i, j] += np.einsum('abc,dbc->ad', m3, H2[v, o, v, v][:, k, :, :])
#                 X2[:, :, i, :] -= np.einsum('abc,lc->abl', m3, H2[o, o, o, v][j, k, :, :])
#     # Apply (ij)(ab) symmetrizer
#     X2 += X2.transpose(1, 0, 3, 2)
#     return X1, X2