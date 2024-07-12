import time
import numpy as np
from miniccpy.utilities import get_memory_usage
from miniccpy.lib import dipeom4_star_p

def spin_to_spatial(x):
    if x % 2 == 1:
        return x // 2 + 1
    else:
        return x // 2

def kernel(T, R, L, omega, fock, g, H1, H2, o, v):
    # Compute the noniterative correction to DIP(3h-1p) energies
    t1, t2 = T
    r1, r2 = R
    deltaA = calc_dipeom4star(r1, r2, t1, t2, fock, g, omega, H1, H2, o, v)
    delta = {"A": deltaA, "B": 0.0, "C": 0.0, "D": 0.0}
    return delta

def build_HR_intermediates(r1, r2, t1, t2, H1, H2, o, v):
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
    )
    # antisymmetrize A(ij)
    I_oovv -= np.transpose(I_oovv, (1, 0, 2, 3))
    return I_oooo, I_oovv

def build_HR3(r2, t2, g, I_oooo, I_oovv, o, v): 
    # Moment-like terms
    X3 = (4.0 / 48.0) * np.einsum("dcle,ijek->ijcdkl", g[v, v, o, v], r2, optimize=True) # T2
    X3 -= (12.0 / 48.0) * np.einsum("dmlk,ijcm->ijcdkl", g[v, o, o, o], r2, optimize=True) # T2
    X3 -= (4.0 / 48.0) * np.einsum("ijmk,cdml->ijcdkl", I_oooo, t2, optimize=True)
    X3 += (12.0 / 48.0) * np.einsum("ijce,edkl->ijcdkl", I_oovv, t2, optimize=True)
    # antisymmetrize A(ijkl)A(cd)
    X3 -= np.transpose(X3, (0, 1, 3, 2, 4, 5)) # A(cd)
    X3 -= np.transpose(X3, (0, 4, 2, 3, 1, 5)) # A(jk)
    X3 -= np.transpose(X3, (1, 0, 2, 3, 4, 5)) + np.transpose(X3, (4, 1, 2, 3, 0, 5)) # A(i/jk)
    X3 -= np.transpose(X3, (5, 1, 2, 3, 4, 0)) + np.transpose(X3, (0, 5, 2, 3, 4, 1)) + np.transpose(X3, (0, 1, 2, 3, 5, 4)) # A(l/ijk)
    return X3

def calc_dipeom4star(r1, r2, t1, t2, fock, g, omega, H1, H2, o, v):
    """Compute the projection of HR on 4h-2p excitations
        X[i, j, c, d, k, l] = < ijklcd | [ HBar(CCSD) * (R1 + R2 + R3) ]_C | 0 >,
        approximated complete to 3rd-order in MBPT (assuming 2h is 0th order). The
        resulting terms include (H[2]*R1)_C + (H[1]*R2)_C + (F_N*R3)_C.
    """

    I_oooo, I_oovv = build_HR_intermediates(r1, r2, t1, t2, H1, H2, o, v)

    #delta_star = dipeom4_star_p.dipeom4_star_p.build_hr4_p_noniterative(
    #        t2, r2, omega,
    #        fock[o, o], fock[v, v],
    #        g[v, v, o, v], g[v, o, o, o], I_oooo, I_oovv,
    #)

    nu, no = t1.shape
    eps = np.diagonal(fock)
    n = np.newaxis
    e_ijcdkl = (-eps[o, n, n, n, n, n] - eps[n, o, n, n, n, n] + eps[n, n, v, n, n, n] + eps[n, n, n, v, n, n] - eps[n, n, n, n, o, n] - eps[n, n, n, n, n, o])
    X3 = build_HR3(r2, t2, g, I_oooo, I_oovv, o, v) 
    L3 = X3/(omega - e_ijcdkl)
    delta_star = (1.0 / 48.0) * np.einsum("ijcdkl,ijcdkl->", L3, X3, optimize=True)


    # get abaa
    #X3B = np.zeros((no // 2, no // 2, nu // 2, nu // 2, no // 2, no // 2))
    #for i in range(no):
    #    isp = spin_to_spatial(i)
    #    for j in range(i + 1, no):
    #        jsp = spin_to_spatial(j)
    #        if i % 2 == 0 and j % 2 == 1:
    #            for k in range(j + 1, no):
    #                ksp = spin_to_spatial(k)
    #                for l in range(k + 1, no):
    #                    lsp = spin_to_spatial(l)
    #                    if k % 2 == 0 and l % 2 == 0:
    #                        for c in range(nu):
    #                            csp = spin_to_spatial(c)
    #                            for d in range(c + 1, nu):
    #                                dsp = spin_to_spatial(d)
    #                                if c % 2 == 0 and d % 2 == 0:
    #                                    X3B[isp, jsp, csp, dsp, ksp, lsp] = X3[i, j, c, d, k, l]
    #                                    X3B[isp, jsp, csp, dsp, lsp, ksp] = -X3[i, j, c, d, k, l]
    #                                    X3B[ksp, jsp, csp, dsp, lsp, isp] = X3[i, j, c, d, k, l]
    #                                    X3B[ksp, jsp, csp, dsp, isp, lsp] = -X3[i, j, c, d, k, l]
    #                                    X3B[lsp, jsp, csp, dsp, isp, ksp] = X3[i, j, c, d, k, l]
    #                                    X3B[lsp, jsp, csp, dsp, ksp, isp] = -X3[i, j, c, d, k, l]
    #                                    #
    #                                    X3B[isp, jsp, dsp, csp, ksp, lsp] = -X3[i, j, c, d, k, l]
    #                                    X3B[isp, jsp, dsp, csp, lsp, ksp] = X3[i, j, c, d, k, l]
    #                                    X3B[ksp, jsp, dsp, csp, lsp, isp] = -X3[i, j, c, d, k, l]
    #                                    X3B[ksp, jsp, dsp, csp, isp, lsp] = X3[i, j, c, d, k, l]
    #                                    X3B[lsp, jsp, dsp, csp, isp, ksp] = -X3[i, j, c, d, k, l]
    #                                    X3B[lsp, jsp, dsp, csp, ksp, isp] = X3[i, j, c, d, k, l]
    #
    #                                    if abs(X3B[isp, jsp, csp, dsp, ksp, lsp]) > 1.0e-03:
    #                                        print(isp, jsp, ksp, lsp, csp, dsp, ":", X3B[isp, jsp, csp, dsp, ksp, lsp])
    #print("norm of abaa = ", np.linalg.norm(X3B.flatten()))

    return delta_star
