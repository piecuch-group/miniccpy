import numpy as np

def update_t2(t2, residual, f, g, o, v, shift, quasi=False):
    
    nu, _, no, _ = t2.shape

    # Quaslinearized update
    if quasi:
        # c(mi) = 1/2 v(mnef) * t2(efin) -> c(ii) = 1/2 v(inef) * t2(efin)
        d1v = np.zeros(nu)
        for a in range(nu):
            d1v[a] = np.einsum("mnf,fmn->", g[o, o, v, v][:, :, a, :], t2[a, :, :, :], optimize=True)
        # c(ae) = 1/2 v(mnef) * t2(afmn)
        d1o = np.zeros(no)
        for i in range(no):
            d1o[i] = np.einsum("nef,efn->", g[o, o, v, v][i, :, :, :], t2[:, :, i, :], optimize=True)
        # c(mnij) = 1/2 v(mnef) * t2(efij)
        d2o = np.zeros((no, no))
        for i in range(no):
            for j in range(no):
                d2o[i, j] = 0.5 * np.einsum("ef,ef->", g[o, o, v, v][i, j, :, :], t2[:, :, i, j], optimize=True)
        # c(abef) = 1/2 v(mnef) * t2(abmn)
        d2v = np.zeros((nu, nu))
        for a in range(nu):
            for b in range(nu):
                d2v[a, b] = 0.5 * np.einsum("mn,mn->", g[o, o, v, v][:, :, a, b], t2[a, b, :, :], optimize=True)
        # c(amie) = v(mnef) * t2(afin)
        d2vo = np.zeros((nu, no))
        for a in range(nu):
            for i in range(no):
                d2vo[a, i] = np.einsum("me,em->", g[o, o, v, v][i, :, a, :], t2[a, :, i, :], optimize=True)
        d3v = np.zeros((nu, no, nu))
        d3o = np.zeros((nu, no, no))
        for a in range(nu):
            for i in range(no):
                for b in range(nu):
                    d3v[a, i, b] = np.dot(g[o, o, v, v][i, :, a, b].T, t2[a, b, i, :])
                for j in range(no):
                    d3o[a, i, j] = np.dot(g[o, o, v, v][i, j, a, :].T, t2[a, :, i, j])

    for a in range(nu):
        for b in range(a + 1, nu):
            for i in range(no):
                for j in range(i + 1, no):
                    denom = f[o, o][i, i] + f[o, o][j, j] - f[v, v][a, a] - f[v, v][b, b]
                    if quasi:
                        denom += 0.5 * ( d1o[i] + d1o[j] + d1v[a] + d1v[b]
                                         - d2vo[a, i] - d2vo[a, j] - d2vo[b, i] - d2v[b, j]
                                         - d2o[i, j] - d2v[a, b]
                                         + d3v[a, i, b] + d3v[a, j, b]
                                         + d3o[a, i, j] + d3o[b, i, j]
                        )
                    
                    t2[a, b, i, j] += residual[a, b, i, j]/(denom - shift)
                    t2[a, b, j, i] = -t2[a, b, i, j]
                    t2[b, a, i, j] = -t2[a, b, i, j]
                    t2[b, a, j, i] = t2[a, b, i, j]
    return t2
