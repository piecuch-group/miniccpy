import numpy as np

def get_3body_hbar_triples_diagonal(g_oovv, t2):
    """< ijkabc | (V_V*T2)_C | ijkabc > diagonal"""

    nu, _, no, _ = t2.shape

    d3_fcn_v = lambda a, i, b: -np.dot(g_oovv[i, :, a, b].T, t2[a, b, i, :])
    d3_fcn_o = lambda a, i, j: np.dot(g_oovv[i, j, a, :].T, t2[a, :, i, j])

    d3v = np.zeros((nu, no, nu))
    d3o = np.zeros((nu, no, no))

    for a in range(nu):
        for i in range(no):
            for b in range(nu):
                d3v[a, i, b] = d3_fcn_v(a, i, b)
    for a in range(nu):
        for i in range(no):
            for j in range(no):
                d3o[a, i, j] = d3_fcn_o(a, i, j)

    return d3v, d3o

def vv_denom_abc(fock, v):
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
