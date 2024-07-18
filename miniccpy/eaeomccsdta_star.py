import numpy as np
from miniccpy.hbar_diagonal import vv_denom_abc

def kernel(T, R, L, omega, fock, g, H1, H2, o, v):
    '''
    '''
    t1, t2 = T
    r1, r2 = R
    if L is None:
        use_L = False
    else:
        l1, l2 = L
        use_L = True

    # orbital dimensions
    nu, no = t1.shape
    # MP denominator
    eps = np.diagonal(fock)
    n = np.newaxis
    e_abcjk = (eps[v, n, n, n, n] + eps[n, v, n, n, n] + eps[n, n, v, n, n]
               - eps[n, n, n, o, n] - eps[n, n, n, n, o])

    # Intermediates
    I2_voo = (
                -np.einsum("amje,e->amj", g[v, o, o, v], r1, optimize=True)
    )
    I2_vvv = (
                0.5 * np.einsum("bcef,e->bcf", g[v, v, v, v], r1, optimize=True)
    )
    I2_vvv -= np.transpose(I2_vvv, (1, 0, 2))

    # Excited-state moment
    M3 = -(3.0 / 12.0) * np.einsum("cmkj,abm->abcjk", g[v, o, o, o], r2, optimize=True)
    M3 += (6.0 / 12.0) * np.einsum("cbke,aej->abcjk", g[v, v, o, v], r2, optimize=True)
    M3 -= (6.0 / 12.0) * np.einsum("amj,bcmk->abcjk", I2_voo, t2, optimize=True)
    M3 += (3.0 / 12.0) * np.einsum("abe,ecjk->abcjk", I2_vvv, t2, optimize=True)
    # antisymmetrize
    M3 -= np.transpose(M3, (1, 0, 2, 3, 4)) + np.transpose(M3, (2, 1, 0, 3, 4))  # antisymmetrize A(a/bc)
    M3 -= np.transpose(M3, (0, 2, 1, 3, 4))  # antisymmetrize A(bc)
    M3 -= np.transpose(M3, (0, 1, 2, 4, 3))  # antisymmetrize A(jk)

    # Left
    if use_L:
        # moment-like terms < 0 | (L1p+L2p1h)*(H_N e^(T1+T2))_C | jkabc >
        L3 = (3.0 / 12.0) * np.einsum("a,jkbc->abcjk", l1, g[o, o, v, v], optimize=True)
        # L3 += (6.0 / 12.0) * np.einsum("abj,kc->abcjk", l2, H1[o, v], optimize=True)
        L3 -= (3.0 / 12.0) * np.einsum("abm,jkmc->abcjk", l2, g[o, o, o, v], optimize=True)
        L3 += (6.0 / 12.0) * np.einsum("eck,ejab->abcjk", l2, g[v, o, v, v], optimize=True)
        # antisymmetrize
        L3 -= np.transpose(L3, (1, 0, 2, 3, 4)) + np.transpose(L3, (2, 1, 0, 3, 4))  # antisymmetrize A(a/bc)
        L3 -= np.transpose(L3, (0, 2, 1, 3, 4))  # antisymmetrize A(bc)
        L3 -= np.transpose(L3, (0, 1, 2, 4, 3))  # antisymmetrize A(jk)
    else:
        L3 = M3

    # divide L3 by MP denominator
    L3 /= (omega - e_abcjk)

    # contract L3*M3
    delta_A = (1.0 / 12.0) * np.einsum("abcjk,abcjk->", L3, M3, optimize=True)

    # Normalize by <R|R> if biorthogonal left vector is not used
    if not use_L:
        rnorm = np.einsum("a,a->", r1, r1, optimize=True)
        rnorm += 0.5 * np.einsum("abj,abj->", r2, r2, optimize=True)
        delta_A /= rnorm

    # Store triples corrections in dictionary
    delta_T = {"A": delta_A, "B": 0.0, "C": 0.0, "D": 0.0}

    return delta_T