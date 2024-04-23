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
