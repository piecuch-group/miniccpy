import numpy as np

def kernel(T, L, f, H1, g, o, v):
    from miniccpy.energy import cc_energy

    t1, t2 = T
    nu, no = t1.shape

    # orbital denominators
    eps = np.diagonal(f)
    n = np.newaxis
    e_abc = -eps[v, n, n] - eps[n, v, n] - eps[n, n, v]
    e_abij = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
    e_ai = 1.0 / (-eps[v, n] + eps[n, o])

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
    t1_corr = singles_res * e_ai
    t2_corr = doubles_res * e_abij

    # base part
    delta_A = cc_energy(t1_corr, t2_corr, f, g, o, v)
    # add in product terms
    delta_A += 0.5 * np.einsum("mnef,em,fn->", g[o, o, v, v], t1, t1_corr)
    delta_A += 0.5 * np.einsum("mnef,em,fn->", g[o, o, v, v], t1_corr, t1)
    delta_B = 0.0
    delta_C = 0.0
    delta_D = 0.0

    # Store triples corrections in dictionary
    delta_T = {"A": delta_A, "B": delta_B, "C": delta_C, "D": delta_D}
    return delta_T