import numpy as np

def kernel(T, L, H1, H2, o, v):

    t1, t2 = T
    l1, l2 = L
    eps = np.diagonal(H1)
    n = np.newaxis
    e_abcijk = (- eps[v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o])
    
    I_vooo = H2[v, o, o, o] - np.einsum("me,aeij->amij", H1[o, v], t2, optimize=True)
    M3 = (9.0 / 36.0) * (
             np.einsum("abie,ecjk->abcijk", H2[v, v, o, v], t2, optimize=True)
            -np.einsum("amij,bcmk->abcijk", I_vooo, t2, optimize=True)
    )
    M3 -= np.transpose(M3, (0, 1, 2, 3, 5, 4)) # (jk)
    M3 -= np.transpose(M3, (0, 1, 2, 4, 3, 5)) + np.transpose(M3, (0, 1, 2, 5, 4, 3)) # (i/jk)
    M3 -= np.transpose(M3, (0, 2, 1, 3, 4, 5)) # (bc)
    M3 -= np.transpose(M3, (2, 1, 0, 3, 4, 5)) + np.transpose(M3, (1, 0, 2, 3, 4, 5)) # (a/bc)

    L3 = (9.0 / 36.0) * (
            np.einsum("ijab,ck->abcijk", H2[o, o, v, v], l1, optimize=True)
           +np.einsum("ia,bcjk->abcijk", H1[o, v], l2, optimize=True)
           +np.einsum("eiba,ecjk->abcijk", H2[v, o, v, v], l2, optimize=True)
           -np.einsum("jima,bcmk->abcijk", H2[o, o, o, v], l2, optimize=True)
    )
    L3 -= np.transpose(L3, (0, 1, 2, 3, 5, 4)) # (jk)
    L3 -= np.transpose(L3, (0, 1, 2, 4, 3, 5)) + np.transpose(L3, (0, 1, 2, 5, 4, 3)) # (i/jk)
    L3 -= np.transpose(L3, (0, 2, 1, 3, 4, 5)) # (bc)
    L3 -= np.transpose(L3, (2, 1, 0, 3, 4, 5)) + np.transpose(L3, (1, 0, 2, 3, 4, 5)) # (a/bc)
    L3 /= e_abcijk

    return (1.0 / 36.0) * np.einsum("abcijk,abcijk->", L3, M3, optimize=True)
