import time
import numpy as np
from miniccpy.utilities import get_memory_usage

def kernel(T, R, L, omega, fock, g, H1, H2, o, v):
    # Compute the noniterative correction to DIP(3h-1p) energies
    t1, t2 = T
    r1, r2 = R
    l1, l2 = L
    eps = np.diagonal(fock)
    n = np.newaxis
    e_ijcdkl = (-eps[o, n, n, n, n, n] - eps[n, o, n, n, n, n] + eps[n, n, v, n, n, n] + eps[n, n, n, v, n, n] - eps[n, n, n, n, o, n] - eps[n, n, n, n, n, o])

    print("    ==> DIP-EOMCC(3h-1p,4h-2p) moment correction <==")

    m4 = build_moments_4h2p(r1, r2, t1, t2, H1, H2, o, v)
    l4 = build_leftamps_4h2p(l1, l2, t1, t2, H1, H2, o, v)
    l4 /= (omega - e_ijcdkl)

    deltaA = (1.0 / 48.0) * np.einsum("ijcdkl,ijcdkl->", l4, m4, optimize=True)

    delta = {"A": deltaA, "B": 0.0, "C": 0.0, "D": 0.0}
    return delta

def build_moments_4h2p(r1, r2, t1, t2, H1, H2, o, v):
    """Compute the projection of HR on 4h-2p excitations
        X[i, j, c, d, k, l] = < ijklcd | [ HBar(CCSD) * (R1 + R2) ]_C | 0 >.
    """
    # Intermediates
    # This term factorizes the 4-body HBar formed by (V_N*T2^2)_C, which enters at 3rd-order.
    # This should be removed too
    I_vv = (
            0.5 * np.einsum("mnef,mn->ef", H2[o, o, v, v], r1, optimize=True)
    )

    # I(ijmk)
    I_oooo = (
          (3.0 / 6.0) * np.einsum("nmke,ijem->ijnk", H2[o, o, o, v], r2, optimize=True) # includes T1
        - (3.0 / 6.0) * np.einsum("mnik,mj->ijnk", H2[o, o, o, o], r1, optimize=True) # includes T1 and T2
    )
    # antisymmetrize A(ijk)
    I_oooo -= np.transpose(I_oooo, (0, 3, 2, 1)) # A(jk)
    I_oooo -= np.transpose(I_oooo, (1, 0, 2, 3)) + np.transpose(I_oooo, (3, 1, 2, 0)) # A(i/jk)

    # I(ijce)
    I_oovv = (
        (1.0 / 2.0) * np.einsum("cmfe,ijem->ijcf", H2[v, o, v, v], r2, optimize=True) # includes T1
        + np.einsum("bmje,mk->jkbe", H2[v, o, o, v], r1, optimize=True) # includes T1 and T2
        + 0.5 * np.einsum("nmie,njcm->ijce", H2[o, o, o, v], r2, optimize=True) # includes T1
        + 0.25 * np.einsum("ef,edil->lidf", I_vv, t2, optimize=True) # remove 4-body HBar term
    )
    # antisymmetrize A(ij)
    I_oovv -= np.transpose(I_oovv, (1, 0, 2, 3))

    # Moment-like terms
    X3 = (4.0 / 48.0) * np.einsum("dcle,ijek->ijcdkl", H2[v, v, o, v], r2, optimize=True) # T2
    X3 -= (12.0 / 48.0) * np.einsum("dmlk,ijcm->ijcdkl", H2[v, o, o, o], r2, optimize=True) # T2
    X3 -= (4.0 / 48.0) * np.einsum("ijmk,cdml->ijcdkl", I_oooo, t2, optimize=True)
    X3 += (12.0 / 48.0) * np.einsum("ijce,edkl->ijcdkl", I_oovv, t2, optimize=True)
    # antisymmetrize A(ijkl)A(cd)
    X3 -= np.transpose(X3, (0, 1, 3, 2, 4, 5)) # A(cd)
    X3 -= np.transpose(X3, (0, 4, 2, 3, 1, 5)) # A(jk)
    X3 -= np.transpose(X3, (1, 0, 2, 3, 4, 5)) + np.transpose(X3, (4, 1, 2, 3, 0, 5)) # A(i/jk)
    X3 -= np.transpose(X3, (5, 1, 2, 3, 4, 0)) + np.transpose(X3, (0, 5, 2, 3, 4, 1)) + np.transpose(X3, (0, 1, 2, 3, 5, 4)) # A(l/ijk)
    return X3

def build_leftamps_4h2p(l1, l2, t1, t2, H1, H2, o, v):
    # Moment-like terms
    X3 = (6.0 / 48.0) * np.einsum("ij,klcd->ijcdkl", l1, H2[o, o, v, v], optimize=True)
    X3 += (8.0 / 48.0) * np.einsum("ijck,ld->ijcdkl", l2, H1[o, v], optimize=True)
    X3 += (4.0 / 48.0) * np.einsum("elcd,ijek->ijcdkl", H2[v, o, v, v], l2, optimize=True)
    X3 -= (12.0 / 48.0) * np.einsum("klmd,ijcm->ijcdkl", H2[o, o, o, v], l2, optimize=True)
    # antisymmetrize A(ijkl)A(cd)
    X3 -= np.transpose(X3, (0, 1, 3, 2, 4, 5)) # A(cd)
    X3 -= np.transpose(X3, (0, 4, 2, 3, 1, 5)) # A(jk)
    X3 -= np.transpose(X3, (1, 0, 2, 3, 4, 5)) + np.transpose(X3, (4, 1, 2, 3, 0, 5)) # A(i/jk)
    X3 -= np.transpose(X3, (5, 1, 2, 3, 4, 0)) + np.transpose(X3, (0, 5, 2, 3, 4, 1)) + np.transpose(X3, (0, 1, 2, 3, 5, 4)) # A(l/ijk)
    return X3
