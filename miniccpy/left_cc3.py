import time
import numpy as np
from miniccpy.energy import lccsd_energy as lcc_energy
from miniccpy.diis import DIIS
from miniccpy.utilities import get_memory_usage
from miniccpy.helper_cc3 import compute_leftcc3_intermediates, get_lr_intermediates

def LH_singles(l1, l2, t1, t2, H1, H2, X1, X2, h_voov, h_vvvv, h_oooo, o, v):
    """Compute the projection of the CCSD Hamiltonian on singles
        X[a, i] = < 0 | (1 + L1 + L2)*(H_N exp(T1+T2))_C | 0 >
    """
    LH = np.einsum("ea,ei->ai", H1[v, v], l1, optimize=True)
    LH -= np.einsum("im,am->ai", H1[o, o], l1, optimize=True)
    LH += np.einsum("eima,em->ai", H2[v, o, o, v], l1, optimize=True)
    LH += 0.5 * np.einsum("fena,efin->ai", H2[v, v, o, v], l2, optimize=True)
    LH -= 0.5 * np.einsum("finm,afmn->ai", H2[v, o, o, o], l2, optimize=True)

    I1 = 0.25 * np.einsum("efmn,fgnm->ge", l2, t2, optimize=True)
    I2 = -0.25 * np.einsum("efmn,egnm->gf", l2, t2, optimize=True)
    I3 = -0.25 * np.einsum("efmo,efno->mn", l2, t2, optimize=True)
    I4 = 0.25 * np.einsum("efmo,efnm->on", l2, t2, optimize=True)
    LH += np.einsum("ge,eiga->ai", I1, H2[v, o, v, v], optimize=True)
    LH += np.einsum("gf,figa->ai", I2, H2[v, o, v, v], optimize=True)
    LH += np.einsum("mn,nima->ai", I3, H2[o, o, o, v], optimize=True)
    LH += np.einsum("on,nioa->ai", I4, H2[o, o, o, v], optimize=True)

    # < 0 | L2 * (H(2) * T3)_C | ia >
    LH += np.einsum("em,imae->ai", X1["vo"], H2[o, o, v, v], optimize=True)
    LH += 0.5 * np.einsum("nmoa,iomn->ai", X2["ooov"], h_oooo, optimize=True)
    LH += np.einsum("fmae,eimf->ai", X2["vovv"], h_voov, optimize=True)
    LH -= 0.5 * np.einsum("gife,efag->ai", X2["vovv"], h_vvvv, optimize=True)
    LH -= np.einsum("imne,enma->ai", X2["ooov"], h_voov, optimize=True)
    LH += H1[o, v].transpose(1, 0)
    return LH

def LH_doubles(l1, l2, t1, t2, f, H1, H2, X1, X2, h_vvov, h_vooo, e_abc, o, v):
    """Compute the projection of the CCSD Hamiltonian on doubles
        X[a, b, i, j] = < ijab | (H_N exp(T1+T2))_C | 0 >
    """
    LH = 0.5 * np.einsum("ea,ebij->abij", H1[v, v], l2, optimize=True)
    LH -= 0.5 * np.einsum("im,abmj->abij", H1[o, o], l2, optimize=True)
    LH += np.einsum("jb,ai->abij", H1[o, v], l1, optimize=True)
    I1 = (
          -0.5 * np.einsum("afmn,efmn->ea", l2, t2, optimize=True)
    )
    LH += 0.5 * np.einsum("ea,ijeb->abij", I1, H2[o, o, v, v], optimize=True)
    I1 = (
          0.5 * np.einsum("efin,efmn->im", l2, t2, optimize=True)
    )
    LH -= 0.5 * np.einsum("im,mjab->abij", I1, H2[o, o, v, v], optimize=True)
    LH += np.einsum("eima,ebmj->abij", H2[v, o, o, v], l2, optimize=True)
    LH += 0.125 * np.einsum("ijmn,abmn->abij", H2[o, o, o, o], l2, optimize=True)
    LH += 0.125 * np.einsum("efab,efij->abij", H2[v, v, v, v], l2, optimize=True)
    LH += 0.5 * np.einsum("ejab,ei->abij", H2[v, o, v, v], l1, optimize=True)
    LH -= 0.5 * np.einsum("ijmb,am->abij", H2[o, o, o, v], l1, optimize=True)

    # Moment-like terms
    #LH += 0.25 * np.einsum("ebfijn,fena->abij", l3, h_vvov, optimize=True) # 1
    #LH -= 0.25 * np.einsum("abfmjn,finm->abij", l3, h_vooo, optimize=True) # 3
    nu, no = l1.shape
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                # fock denominator for occupied
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                l3_abc = 0.5 * (
                        np.einsum("eba,ec->abc", H2[v, o, v, v][:, i, :, :], l2[:, :, j, k], optimize=True)
                        -np.einsum("eba,ec->abc", H2[v, o, v, v][:, j, :, :], l2[:, :, i, k], optimize=True)
                        -np.einsum("eba,ec->abc", H2[v, o, v, v][:, k, :, :], l2[:, :, j, i], optimize=True)
                )
                l3_abc -= 0.5 * (
                        np.einsum("ma,bcm->abc", H2[o, o, o, v][j, i, :, :], l2[:, :, :, k], optimize=True)
                        -np.einsum("ma,bcm->abc", H2[o, o, o, v][k, i, :, :], l2[:, :, :, j], optimize=True)
                        -np.einsum("ma,bcm->abc", H2[o, o, o, v][j, k, :, :], l2[:, :, :, i], optimize=True)
                )
                l3_abc += 0.5 * (
                        np.einsum("ab,c->abc", H2[o, o, v, v][i, j, :, :], l1[:, k], optimize=True)
                        -np.einsum("ab,c->abc", H2[o, o, v, v][k, j, :, :], l1[:, i], optimize=True)
                        -np.einsum("ab,c->abc", H2[o, o, v, v][i, k, :, :], l1[:, j], optimize=True)
                )
                l3_abc += 0.5 * (
                        np.einsum("a,bc->abc", H1[o, v][i, :], l2[:, :, j, k], optimize=True)
                        -np.einsum("a,bc->abc", H1[o, v][j, :], l2[:, :, i, k], optimize=True)
                        -np.einsum("a,bc->abc", H1[o, v][k, :], l2[:, :, j, i], optimize=True)
                )
                # antisymmetrize A(abc)
                l3_abc -= np.transpose(l3_abc, (1, 0, 2)) + np.transpose(l3_abc, (2, 1, 0)) # (a/bc)
                l3_abc -= np.transpose(l3_abc, (0, 2, 1)) # (bc)
                # Divide l_abc by the denominator
                l3_abc /= (denom_occ + e_abc)
                #l3_abc = l3[:, :, :, i, j, k].copy()
                # X2(abij) = 1/2 A(k/ij) l3(efbijk) * h_vvov(feka)
                LH[:, :, i, j] += 0.5 * np.einsum("ebf,fea->ab", l3_abc, h_vvov[:, :, k, :], optimize=True)
                LH[:, :, j, k] += 0.5 * np.einsum("ebf,fea->ab", l3_abc, h_vvov[:, :, i, :], optimize=True)
                LH[:, :, i, k] -= 0.5 * np.einsum("ebf,fea->ab", l3_abc, h_vvov[:, :, j, :], optimize=True)
                # X2(abij) = -1/2 A(j/ik) l3(abfijk) * h_vooo(f:ki)
                LH[:, :, :, j] -= 0.5 * np.einsum("abf,fi->abi", l3_abc, h_vooo[:, :, k, i], optimize=True)
                LH[:, :, :, i] += 0.5 * np.einsum("abf,fi->abi", l3_abc, h_vooo[:, :, k, j], optimize=True)
                LH[:, :, :, k] += 0.5 * np.einsum("abf,fi->abi", l3_abc, h_vooo[:, :, j, i], optimize=True)

    LH += 0.25 * H2[o, o, v, v].transpose(2, 3, 0, 1)
    LH -= np.transpose(LH, (1, 0, 2, 3))
    LH -= np.transpose(LH, (0, 1, 3, 2))
    # Manually clear all diagonal elements
    for a in range(nu):
        LH[a, a, :, :] *= 0.0
    for i in range(no):
        LH[:, :, i, i] *= 0.0
    return LH

def kernel(T, fock, g, H1, H2, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core):
    """Solve the CCSDT system of nonlinear equations using Jacobi iterations
    with DIIS acceleration. The initial values of the T amplitudes are taken to be 0."""

    omega = 0.0
    eps = np.diagonal(fock)
    n = np.newaxis
    e_abc = -eps[v, n, n] - eps[n, v, n] - eps[n, n, v]
    e_abij = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o] - omega + energy_shift)
    e_ai = 1.0 / (-eps[v, n] + eps[n, o] - omega + energy_shift)

    nunocc, nocc = e_ai.shape
    n1 = nocc * nunocc
    n2 = nocc ** 2 * nunocc ** 2
    ndim = n1 + n2

    diis_engine = DIIS(ndim, diis_size, out_of_core)

    # unpack T vector
    t1, t2 = T
    l1 = t1.copy()
    l2 = t2.copy()
    lh1 = np.zeros((nunocc, nocc))
    lh2 = np.zeros((nunocc, nunocc, nocc, nocc))

    old_energy = lcc_energy(l1, l2, lh1, lh2) + omega
    # Get CCS intermediates (it would be nice to not have to recompute these in left-CC)
    h_vvov, h_vooo, h_voov, h_vvvv, h_oooo = compute_leftcc3_intermediates(t1, t2, fock, g, o, v)

    print("    ==> Left-CC3 amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dL|     Wall Time     Memory")
    for idx in range(maxit):

        tic = time.time()

        # comptute L*T intermediates
        X1, X2 = get_lr_intermediates(l1, l2, t2, fock, H1, H2, h_vvov, h_vooo, omega, e_abc, o, v)
        lh1 = LH_singles(l1, l2, t1, t2, H1, H2, X1, X2, h_voov, h_vvvv, h_oooo, o, v)
        lh2 = LH_doubles(l1, l2, t1, t2, fock, H1, H2, X1, X2, h_vvov, h_vooo, e_abc, o, v)

        lh1 = (omega * l1 - lh1) * e_ai
        lh2 = (omega * l2 - lh2) * e_abij
        l1 += lh1
        l2 += lh2

        res_norm = np.linalg.norm(lh1.flatten()) + np.linalg.norm(lh2.flatten())
        current_energy = lcc_energy(l1, l2, lh1, lh2) + omega
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < convergence and res_norm < convergence:
            break

        if idx >= n_start_diis:
            diis_engine.push((l1, l2), (lh1, lh2), idx)
        if idx >= diis_size + n_start_diis:
            L_extrap = diis_engine.extrapolate()
            l1 = L_extrap[:n1].reshape((nunocc, nocc))
            l2 = L_extrap[n1:].reshape((nunocc, nunocc, nocc, nocc))

        old_energy = current_energy

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(idx, current_energy, delta_e, res_norm, minutes, seconds, get_memory_usage()))
                                                                                      res_norm, minutes, seconds))
    else:
        raise ValueError("left-CC3 iterations did not converge")

    diis_engine.cleanup()
    e_corr = lcc_energy(l1, l2, lh1, lh2) + omega

    return (l1, l2), e_corr
