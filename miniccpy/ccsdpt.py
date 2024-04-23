import numpy as np

def kernel(T, L, fock, H1, g, o, v):

    # Note: H1 should just be None. It's not even used. It's just there
    # to make the call in run_correction the same for CCSD(T) as for CR-CC(2,3).

    t1, t2 = T
    eps = np.diagonal(fock)
    n = np.newaxis
    e_abcijk = (- eps[v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o])


    M3 = (9.0 / 36.0) * (
             np.einsum("abie,ecjk->abcijk", g[v, v, o, v], t2, optimize=True)
            -np.einsum("amij,bcmk->abcijk", g[v, o, o, o], t2, optimize=True)
    )
    M3 -= np.transpose(M3, (0, 1, 2, 3, 5, 4)) # (jk)
    M3 -= np.transpose(M3, (0, 1, 2, 4, 3, 5)) + np.transpose(M3, (0, 1, 2, 5, 4, 3)) # (i/jk)
    M3 -= np.transpose(M3, (0, 2, 1, 3, 4, 5)) # (bc)
    M3 -= np.transpose(M3, (2, 1, 0, 3, 4, 5)) + np.transpose(M3, (1, 0, 2, 3, 4, 5)) # (a/bc)

    L3 = (9.0 / 36.0) * (
            np.einsum("ijab,ck->abcijk", g[o, o, v, v], t1, optimize=True)
           +np.einsum("ia,bcjk->abcijk", fock[o, v], t2, optimize=True)
    )
    L3 -= np.transpose(L3, (0, 1, 2, 3, 5, 4)) # (jk)
    L3 -= np.transpose(L3, (0, 1, 2, 4, 3, 5)) + np.transpose(L3, (0, 1, 2, 5, 4, 3)) # (i/jk)
    L3 -= np.transpose(L3, (0, 2, 1, 3, 4, 5)) # (bc)
    L3 -= np.transpose(L3, (2, 1, 0, 3, 4, 5)) + np.transpose(L3, (1, 0, 2, 3, 4, 5)) # (a/bc)
    # For CCSD(T), can re-use M3 to compute the (L2*V_N)_C = ((T2)^+ * V_N)_C
    L3 += M3
    L3 /= e_abcijk

    # Compute CCSD(T) correction
    delta_A = (1.0 / 36.0) * np.einsum("abcijk,abcijk->", L3, M3, optimize=True)

    # store results in dictionary
    delta_T = {"A": delta_A, "B": 0.0, "C": 0.0, "D": 0.0}
    return delta_T
