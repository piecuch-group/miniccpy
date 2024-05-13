import time
import numpy as np
from miniccpy.energy import lccsd_energy as lcc_energy
from miniccpy.diis import DIIS
from miniccpy.helper_cc3 import compute_ccs_intermediates

def get_lr_intermediates(l1, l2, l3, t2, f, H1, H2, h_vvov, h_vooo, e_abc, o, v):
    nu, _, no, _ = t2.shape
    X1 = {"vo": []}
    X2 = {"ooov": [], "vovv": []}
    # (L2 * T3)_C
    #X1["vo"] = 0.25 * np.einsum("efmn,aefimn->ai", l2, t3, optimize=True)
    X1["vo"] = np.zeros((nu, no))
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                # fock denominator for occupied
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                # -1/2 A(k/ij)A(abc) I(amij) * t(bcmk)
                t3_abc = -0.5 * np.einsum("am,bcm->abc", h_vooo[:, :, i, j], t2[:, :, :, k], optimize=True)
                t3_abc += 0.5 * np.einsum("am,bcm->abc", h_vooo[:, :, k, j], t2[:, :, :, i], optimize=True)
                t3_abc += 0.5 * np.einsum("am,bcm->abc", h_vooo[:, :, i, k], t2[:, :, :, j], optimize=True)
                # 1/2 A(i/jk)A(abc) I(abie) * t(ecjk)
                t3_abc += 0.5 * np.einsum("abe,ec->abc", h_vvov[:, :, i, :], t2[:, :, j, k], optimize=True)
                t3_abc -= 0.5 * np.einsum("abe,ec->abc", h_vvov[:, :, j, :], t2[:, :, i, k], optimize=True)
                t3_abc -= 0.5 * np.einsum("abe,ec->abc", h_vvov[:, :, k, :], t2[:, :, j, i], optimize=True)
                # Antisymmetrize A(abc)
                t3_abc -= np.transpose(t3_abc, (1, 0, 2)) + np.transpose(t3_abc, (2, 1, 0)) # A(a/bc)
                t3_abc -= np.transpose(t3_abc, (0, 2, 1)) # A(bc)
                # Divide t_abc by the denominator
                t3_abc /= (denom_occ + e_abc)
                # 1/4 A(i/jk) l2(bcjk) * t3(abcijk)
                X1["vo"][:, i] += 0.5 * np.einsum("bc,abc->a", l2[:, :, j, k], t3_abc, optimize=True)
                X1["vo"][:, j] -= 0.5 * np.einsum("bc,abc->a", l2[:, :, i, k], t3_abc, optimize=True)
                X1["vo"][:, k] -= 0.5 * np.einsum("bc,abc->a", l2[:, :, j, i], t3_abc, optimize=True)
    # (L3 * T2)_C
    #X2["ooov"] = 0.5 * np.einsum('aefijn,efmn->jima', l3, t2, optimize=True)
    X2["ooov"] = np.zeros((no, no, no, nu))
    #X2["vovv"] = -0.5 * np.einsum('abfimn,efmn->eiba', l3, t2, optimize=True)
    X2["vovv"] = np.zeros((nu, no, nu, nu))
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
                l3_abc = l3[:, :, :, i, j, k].copy()
                # X2(jima) = 1/2 A(k/ij) l3(aefijk) * t2(efmk)
                X2["ooov"][i, j, :, :] -= 0.5 * np.einsum("aef,efm->ma", l3_abc, t2[:, :, :, k], optimize=True)
                X2["ooov"][j, k, :, :] -= 0.5 * np.einsum("aef,efm->ma", l3_abc, t2[:, :, :, i], optimize=True)
                X2["ooov"][i, k, :, :] += 0.5 * np.einsum("aef,efm->ma", l3_abc, t2[:, :, :, j], optimize=True)
                # X2(eiba) = -1/2 A(i/jk) l3(abfijk) * t2(efjk)
                X2["vovv"][:, i, :, :] -= np.einsum("abf,ef->eba", l3_abc, t2[:, :, j, k], optimize=True)
                X2["vovv"][:, j, :, :] += np.einsum("abf,ef->eba", l3_abc, t2[:, :, i, k], optimize=True)
                X2["vovv"][:, k, :, :] += np.einsum("abf,ef->eba", l3_abc, t2[:, :, j, i], optimize=True)
    # antisymmetrize 
    for i in range(no):
        for j in range(i + 1, no):
            X2["ooov"][j, i, :, :] = -X2["ooov"][i, j, :, :]
    # clear diagonal elements
    for a in range(nu):
        X2["vovv"][:, :, a, a] *= 0.0
    return X1, X2


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

def LH_doubles(l1, l2, l3, t1, t2, f, H1, H2, X1, X2, h_vvov, h_vooo, e_abc, o, v):
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
                l3_abc = l3[:, :, :, i, j, k].copy()
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

def LH_triples(l1, l2, l3, t1, t2, f, g, H1, H2, o, v):
    LH = (3.0 / 36.0) * np.einsum("ea,ebcijk->abcijk", f[v, v], l3, optimize=True)
    LH -= (3.0 / 36.0) * np.einsum("im,abcmjk->abcijk", f[o, o], l3, optimize=True)
    # < 0 | L1 * H(2) | ijkabc >
    LH += (9.0 / 36.0) * np.einsum("ai,jkbc->abcijk", l1, g[o, o, v, v], optimize=True)
    # < 0 | L2 * H(2) | ijkabc >
    LH += (9.0 / 36.0) * np.einsum("bcjk,ia->abcijk", l2, H1[o, v], optimize=True)
    LH += (9.0 / 36.0) * np.einsum("ebij,ekac->abcijk", l2, H2[v, o, v, v], optimize=True)
    LH -= (9.0 / 36.0) * np.einsum("abmj,ikmc->abcijk", l2, H2[o, o, o, v], optimize=True)
    #
    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    LH -= np.transpose(LH, (0, 1, 2, 3, 5, 4))
    LH -= np.transpose(LH, (0, 1, 2, 4, 3, 5)) + np.transpose(LH, (0, 1, 2, 5, 4, 3))
    LH -= np.transpose(LH, (0, 2, 1, 3, 4, 5))
    LH -= np.transpose(LH, (1, 0, 2, 3, 4, 5)) + np.transpose(LH, (2, 1, 0, 3, 4, 5))
    return LH

def kernel(T, fock, g, H1, H2, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core):
    """Solve the CCSDT system of nonlinear equations using Jacobi iterations
    with DIIS acceleration. The initial values of the T amplitudes are taken to be 0."""

    omega = 0.0
    eps = np.diagonal(fock)
    n = np.newaxis
    e_abcijk = 1.0 / (- eps[v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                      + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o] - omega + energy_shift)
    e_abc = -eps[v, n, n] - eps[n, v, n] - eps[n, n, v]
    e_abij = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o] - omega + energy_shift)
    e_ai = 1.0 / (-eps[v, n] + eps[n, o] - omega + energy_shift)

    nunocc, nocc = e_ai.shape
    n1 = nocc * nunocc
    n2 = nocc ** 2 * nunocc ** 2
    n3 = nocc ** 3 * nunocc ** 3
    ndim = n1 + n2 + n3

    diis_engine = DIIS(ndim, diis_size, out_of_core)

    # unpack T vector
    t1, t2, t3 = T
    l1 = t1.copy()
    l2 = t2.copy()
    l3 = t3.copy()
    lh1 = np.zeros((nunocc, nocc))
    lh2 = np.zeros((nunocc, nunocc, nocc, nocc))
    lh3 = np.zeros((nunocc, nunocc, nunocc, nocc, nocc, nocc))

    old_energy = lcc_energy(l1, l2, lh1, lh2) + omega

    print("    ==> left-CC3 amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(maxit):

        tic = time.time()

        # Get CCS intermediates (it would be nice to not have to recompute these in left-CC)
        h_vvov, h_vooo, h_voov, h_vvvv, h_oooo = compute_ccs_intermediates(t1, t2, fock, g, o, v)
        # comptute L*T intermediates
        X1, X2 = get_lr_intermediates(l1, l2, l3, t2, fock, H1, H2, h_vvov, h_vooo, e_abc, o, v)
        lh1 = LH_singles(l1, l2, t1, t2, H1, H2, X1, X2, h_voov, h_vvvv, h_oooo, o, v)
        lh2 = LH_doubles(l1, l2, l3, t1, t2, fock, H1, H2, X1, X2, h_vvov, h_vooo, e_abc, o, v)
        lh3 = LH_triples(l1, l2, l3, t1, t2, fock, g, H1, H2, o, v)

        lh1 = (omega * l1 - lh1) * e_ai
        lh2 = (omega * l2 - lh2) * e_abij
        lh3 = (omega * l3 - lh3) * e_abcijk
        l1 += lh1
        l2 += lh2
        l3 += lh3

        res_norm = np.linalg.norm(lh1.flatten()) + np.linalg.norm(lh2.flatten()) + np.linalg.norm(lh3.flatten())
        current_energy = lcc_energy(l1, l2, lh1, lh2) + omega
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < convergence and res_norm < convergence:
            break

        if idx >= n_start_diis:
            diis_engine.push((l1, l2, l3), (lh1, lh2, lh3), idx)
        if idx >= diis_size + n_start_diis:
            L_extrap = diis_engine.extrapolate()
            l1 = L_extrap[:n1].reshape((nunocc, nocc))
            l2 = L_extrap[n1:n1 + n2].reshape((nunocc, nunocc, nocc, nocc))
            l3 = L_extrap[n1 + n2:].reshape((nunocc, nunocc, nunocc, nocc, nocc, nocc))

        old_energy = current_energy

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(idx, current_energy, delta_e,
                                                                                      res_norm, minutes, seconds))
    else:
        raise ValueError("left-CC3 iterations did not converge")

    diis_engine.cleanup()
    e_corr = lcc_energy(l1, l2, lh1, lh2) + omega

    return (l1, l2, l3), e_corr
