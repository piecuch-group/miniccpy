import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def add_t3_contributions(np.ndarray[np.float64_t, ndim=2] singles_res, 
                         np.ndarray[np.float64_t, ndim=4]  doubles_res,
                         np.ndarray[np.float64_t, ndim=2] t1, 
                         np.ndarray[np.float64_t, ndim=4] t2, 
                         np.ndarray[np.float64_t, ndim=2] f,
                         np.ndarray[np.float64_t, ndim=4] g, 
                         np.ndarray[np.float64_t, ndim=4] I_vooo,
                         np.ndarray[np.float64_t, ndim=4] I_vvov,
                         np.ndarray[np.float64_t, ndim=3] e_abc,
                         o, v):
    cdef int i, j, k, no, nu
    cdef double denom_occ
    cdef np.ndarray[np.float64_t, ndim=3] t3_abc

    # Compute additional CCS-like intermediates
    h_ooov = g[o, o, o, v] + np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    h_vovv = g[v, o, v, v] - np.einsum("mnfe,an->amef", g[o, o, v, v], t1, optimize=True) # no(2)nu(3)
    h_ov = f[o, v] + np.einsum("mnef,fn->me", g[o, o, v, v], t1, optimize=True)
    # get orbital dimensions
    nu = t1.shape[0]
    no = t1.shape[1]
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                # fock denominator for occupied
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                # -1/2 A(k/ij)A(abc) I(amij) * t(bcmk)
                t3_abc = -0.5 * np.einsum("am,bcm->abc", I_vooo[:, :, i, j], t2[:, :, :, k], optimize=True)
                t3_abc += 0.5 * np.einsum("am,bcm->abc", I_vooo[:, :, k, j], t2[:, :, :, i], optimize=True)
                t3_abc += 0.5 * np.einsum("am,bcm->abc", I_vooo[:, :, i, k], t2[:, :, :, j], optimize=True)
                # 1/2 A(i/jk)A(abc) I(abie) * t(ecjk)
                t3_abc += 0.5 * np.einsum("abe,ec->abc", I_vvov[:, :, i, :], t2[:, :, j, k], optimize=True)
                t3_abc -= 0.5 * np.einsum("abe,ec->abc", I_vvov[:, :, j, :], t2[:, :, i, k], optimize=True)
                t3_abc -= 0.5 * np.einsum("abe,ec->abc", I_vvov[:, :, k, :], t2[:, :, j, i], optimize=True)
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
                doubles_res[:, :, i, j] += 0.5 * np.einsum("e,abe->ab", h_ov[k, :], t3_abc, optimize=True) # (1)
                doubles_res[:, :, j, k] += 0.5 * np.einsum("e,abe->ab", h_ov[i, :], t3_abc, optimize=True) # (ik)
                doubles_res[:, :, i, k] -= 0.5 * np.einsum("e,abe->ab", h_ov[j, :], t3_abc, optimize=True) # (jk)
                # Compute diagram: -A(j/ik) h(ik:f) * t3(abfijk)
                doubles_res[:, :, :, j] -= 0.5 * np.einsum("mf,abf->abm", h_ooov[i, k, :, :], t3_abc, optimize=True)
                doubles_res[:, :, :, i] += 0.5 * np.einsum("mf,abf->abm", h_ooov[j, k, :, :], t3_abc, optimize=True)
                doubles_res[:, :, :, k] += 0.5 * np.einsum("mf,abf->abm", h_ooov[i, j, :, :], t3_abc, optimize=True)
                # Compute diagram: 1/2 A(k/ij) h(akef) * t3(ebfijk)
                doubles_res[:, :, i, j] += 0.5 * np.einsum("aef,ebf->ab", h_vovv[:, k, :, :], t3_abc, optimize=True)
                doubles_res[:, :, j, k] += 0.5 * np.einsum("aef,ebf->ab", h_vovv[:, i, :, :], t3_abc, optimize=True)
                doubles_res[:, :, i, k] -= 0.5 * np.einsum("aef,ebf->ab", h_vovv[:, j, :, :], t3_abc, optimize=True)
    # Antisymmetrize
    doubles_res -= np.transpose(doubles_res, (1, 0, 2, 3))
    doubles_res -= np.transpose(doubles_res, (0, 1, 3, 2))
    # Manually clear all diagonal elements
    for a in range(nu):
        doubles_res[a, a, :, :] *= 0.0
    for i in range(no):
        doubles_res[:, :, i, i] *= 0.0
    return singles_res, doubles_res
