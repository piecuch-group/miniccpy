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
    e_ibcjk = (-eps[o, n, n, n, n] + eps[n, v, n, n, n] + eps[n, n, v, n, n]
               - eps[n, n, n, o, n] - eps[n, n, n, n, o])

    # Intermediates
    I2_ooo = (
            - 0.5 * np.einsum("mnji,n->mij", g[o, o, o, o], r1, optimize=True)
    )
    I2_ooo -= np.transpose(I2_ooo, (0, 2, 1))
    I2_ovv = (
            + np.einsum("bmie,m->ibe", g[v, o, o, v], r1, optimize=True)
    )

    # Excited-state moment
    M3 = -(6.0 / 12.0) * np.einsum("cmkj,ibm->ibcjk", g[v, o, o, o], r2, optimize=True)
    M3 += (3.0 / 12.0) * np.einsum("cbke,iej->ibcjk", g[v, v, o, v], r2, optimize=True)
    M3 -= (3.0 / 12.0) * np.einsum("mij,bcmk->ibcjk", I2_ooo, t2, optimize=True)
    M3 += (6.0 / 12.0) * np.einsum("ibe,ecjk->ibcjk", I2_ovv, t2, optimize=True)
    M3 -= np.transpose(M3, (3, 1, 2, 0, 4)) + np.transpose(M3, (4, 1, 2, 3, 0)) # antisymmetrize A(i/jk)
    M3 -= np.transpose(M3, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    M3 -= np.transpose(M3, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)

    # Left
    if use_L:
        # moment-like terms < 0 | (L1p+L2h1p)*(H_N e^(T1+T2))_C | ijkbc >
        L3 = (3.0 / 12.0) * np.einsum("i,jkbc->ibcjk", l1, g[o, o, v, v], optimize=True)
        # L3 += (6.0 / 12.0) * np.einsum("ibj,kc->ibcjk", l2, H1[o, v], optimize=True)
        L3 -= (6.0 / 12.0) * np.einsum("mck,ijmb->ibcjk", l2, g[o, o, o, v], optimize=True)
        L3 += (3.0 / 12.0) * np.einsum("iej,ekbc->ibcjk", l2, g[v, o, v, v], optimize=True)
        L3 -= np.transpose(L3, (3, 1, 2, 0, 4)) + np.transpose(L3, (4, 1, 2, 3, 0)) # antisymmetrize A(i/jk)
        L3 -= np.transpose(L3, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
        L3 -= np.transpose(L3, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    else:
        L3 = M3

    # divide L3 by MP denominator
    L3 /= (omega - e_ibcjk)

    # contract L3*M3
    delta_A = (1.0 / 12.0) * np.einsum("ibcjk,ibcjk->", L3, M3, optimize=True)

    # Normalize by <R|R> if biorthogonal left vector is not used
    if not use_L:
        rnorm = np.einsum("i,i->", r1, r1, optimize=True)
        rnorm += 0.5 * np.einsum("ibj,ibj->", r2, r2, optimize=True)
        delta_A /= rnorm

    # Store triples corrections in dictionary
    delta_T = {"A": delta_A, "B": 0.0, "C": 0.0, "D": 0.0}

    return delta_T
