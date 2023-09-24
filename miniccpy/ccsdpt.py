import numpy as np

def kernel(T, fock, g, o, v):

    t1, t2 = T

    L3 = (9.0 / 36.0) * (
            np.einsum("ijab,ck->abcijk", g[o, o, v, v], t1, optimize=True)
           +np.einsum("ia,bcjk->abcijk", fock[o, v], t2, optimize=True)
           +np.einsum("ejab,ecik->abcijk", g[v, o, v, v], t2, optimize=True)
           -np.einsum("ibmj,acmk->abcijk", g[o, v, o, o], t2, optimize=True)
    )
    L3 -= np.transpose(L3, (0, 1, 2, 3, 5, 4)) # (jk)
    L3 -= np.transpose(L3, (0, 1, 2, 4, 3, 5)) + np.transpose(L3, (0, 1, 2, 5, 4, 3)) # (i/jk)
    L3 -= np.transpose(L3, (0, 2, 1, 3, 4, 5)) # (bc)
    L3 -= np.transpose(L3, (2, 1, 0, 3, 4, 5)) + np.transpose(L3, (1, 0, 2, 3, 4, 5)) # (a/bc)

    M3 = (9.0 / 36.0) * (
             np.einsum("abie,ecjk->abcijk", g[v, v, o, v], t2, optimize=True)
            -np.einsum("amij,bcmk->abcijk", g[v, o, o, o], t2, optimize=True)
    )
    M3 -= np.transpose(M3, (0, 1, 2, 3, 5, 4)) # (jk)
    M3 -= np.transpose(M3, (0, 1, 2, 4, 3, 5)) + np.transpose(M3, (0, 1, 2, 5, 4, 3)) # (i/jk)
    M3 -= np.transpose(M3, (0, 2, 1, 3, 4, 5)) # (bc)
    M3 -= np.transpose(M3, (2, 1, 0, 3, 4, 5)) + np.transpose(M3, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return (1.0 / 36.0) * np.einsum("abcijk,abcijk->", L3, M3, optimize=True)
