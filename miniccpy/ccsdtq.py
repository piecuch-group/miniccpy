import time
import numpy as np
from miniccpy.energy import cc_energy
from miniccpy.helper_cc import get_ccs_intermediates, get_ccsd_intermediates
from miniccpy.diis import DIIS
from miniccpy.utilities import get_memory_usage

def singles_residual(t1, t2, t3, f, g, o, v):
    """Compute the projection of the CCSDTQ Hamiltonian on singles
        X[a, i] = < ia | (H_N exp(T1+T2+T3+T4))_C | 0 >
    """

    chi_vv = f[v, v] + np.einsum("anef,fn->ae", g[v, o, v, v], t1, optimize=True)
    chi_oo = f[o, o] + np.einsum("mnif,fn->mi", g[o, o, o, v], t1, optimize=True)
    h_ov = f[o, v] + np.einsum("mnef,fn->me", g[o, o, v, v], t1, optimize=True)
    h_oo = chi_oo + np.einsum("me,ei->mi", h_ov, t1, optimize=True)
    h_ooov = g[o, o, o, v] + np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    h_vovv = g[v, o, v, v] - np.einsum("mnfe,an->amef", g[o, o, v, v], t1, optimize=True)

    singles_res = -np.einsum("mi,am->ai", h_oo, t1, optimize=True)
    singles_res += np.einsum("ae,ei->ai", chi_vv, t1, optimize=True)
    singles_res += np.einsum("anif,fn->ai", g[v, o, o, v], t1, optimize=True)
    singles_res += np.einsum("me,aeim->ai", h_ov, t2, optimize=True)
    singles_res -= 0.5 * np.einsum("mnif,afmn->ai", h_ooov, t2, optimize=True)
    singles_res += 0.5 * np.einsum("anef,efin->ai", h_vovv, t2, optimize=True)
    singles_res += 0.25 * np.einsum("mnef,aefimn->ai", g[o, o, v, v], t3, optimize=True)
    singles_res += f[v, o]
    return singles_res

def doubles_residual(t1, t2, t3, t4, f, g, o, v):
    """Compute the projection of the CCSDTQ Hamiltonian on doubles
        X[a, b, i, j] = < ijab | (H_N exp(T1+T2+T3+T4))_C | 0 >
    """

    H1, H2 = get_ccs_intermediates(t1, f, g, o, v)

    # intermediates
    I_oo = H1[o, o] + 0.5 * np.einsum("mnef,efin->mi", g[o, o, v, v], t2, optimize=True)
    I_vv = H1[v, v] - 0.5 * np.einsum("mnef,afmn->ae", g[o, o, v, v], t2, optimize=True)
    I_voov = H2[v, o, o, v] + 0.5 * np.einsum("mnef,afin->amie", g[o, o, v, v], t2, optimize=True)
    I_oooo = H2[o, o, o, o] + 0.5 * np.einsum("mnef,efij->mnij", g[o, o, v, v], t2, optimize=True)
    I_vooo = H2[v, o, o, o] + 0.5 * np.einsum('anef,efij->anij', g[v, o, v, v] + 0.5 * H2[v, o, v, v], t2, optimize=True)
    tau = 0.5 * t2 + np.einsum('ai,bj->abij', t1, t1, optimize=True)

    doubles_res = -0.5 * np.einsum("amij,bm->abij", I_vooo, t1, optimize=True)
    doubles_res += 0.5 * np.einsum("abie,ej->abij", H2[v, v, o, v], t1, optimize=True)
    doubles_res += 0.5 * np.einsum("ae,ebij->abij", I_vv, t2, optimize=True)
    doubles_res -= 0.5 * np.einsum("mi,abmj->abij", I_oo, t2, optimize=True)
    doubles_res += np.einsum("amie,ebmj->abij", I_voov, t2, optimize=True)
    doubles_res += 0.25 * np.einsum("abef,efij->abij", g[v, v, v, v], tau, optimize=True)
    doubles_res += 0.125 * np.einsum("mnij,abmn->abij", I_oooo, t2, optimize=True)
    doubles_res += 0.25 * np.einsum("me,abeijm->abij", H1[o, v], t3, optimize=True)
    doubles_res -= 0.25 * np.einsum("mnif,abfmjn->abij", g[o, o, o, v] + H2[o, o, o, v], t3, optimize=True)
    doubles_res += 0.25 * np.einsum("anef,ebfijn->abij", g[v, o, v, v] + H2[v, o, v, v], t3, optimize=True)
    doubles_res += (1.0 / 4.0) * 0.25 * np.einsum("mnef,abefijmn->abij", g[o, o, v, v], t4, optimize=True)

    doubles_res -= np.transpose(doubles_res, (1, 0, 2, 3))
    doubles_res -= np.transpose(doubles_res, (0, 1, 3, 2))
    doubles_res += g[v, v, o, o]
    return doubles_res

def triples_residual(t1, t2, t3, t4, f, g, o, v):
    """Compute the projection of the CCSDTQ Hamiltonian on triples
        X[a, b, c, i, j, k] = < ijkabc | (H_N exp(T1+T2+T3+T4))_C | 0 >
    """

    H1, H2 = get_ccsd_intermediates(t1, t2, f, g, o, v)

    I_vvov = H2[v, v, o, v] + (
              -0.5 * np.einsum("mnef,abfimn->abie", g[o, o, v, v], t3, optimize=True)
              +np.einsum("me,abim->abie", H1[o, v], t2, optimize=True)
    )
    I_vooo = H2[v, o, o, o] + 0.5 * np.einsum("mnef,aefijn->amij", g[o, o, v, v], t3, optimize=True)

    triples_res = -0.25 * np.einsum("amij,bcmk->abcijk", I_vooo, t2, optimize=True)
    triples_res += 0.25 * np.einsum("abie,ecjk->abcijk", I_vvov, t2, optimize=True)
    triples_res -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H1[o, o], t3, optimize=True)
    triples_res += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H1[v, v], t3, optimize=True)
    triples_res += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H2[o, o, o, o], t3, optimize=True)
    triples_res += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H2[v, v, v, v], t3, optimize=True)
    triples_res += 0.25 * np.einsum("cmke,abeijm->abcijk", H2[v, o, o, v], t3, optimize=True)
    triples_res += (1.0 / 36.0) * np.einsum("me,abceijkm->abcijk", H1[o, v], t4, optimize=True) # (1) = 1
    triples_res += (1.0 / 24.0) * np.einsum("cnef,abefijkn->abcijk", H2[v, o, v, v], t4, optimize=True) # (c/ab) = 3
    triples_res -= (1.0 / 24.0) * np.einsum("mnkf,abcfijmn->abcijk", H2[o, o, o, v], t4, optimize=True) # (k/ij) = 3

    triples_res -= np.transpose(triples_res, (0, 1, 2, 3, 5, 4)) # (jk)
    triples_res -= np.transpose(triples_res, (0, 1, 2, 4, 3, 5)) + np.transpose(triples_res, (0, 1, 2, 5, 4, 3)) # (i/jk)
    triples_res -= np.transpose(triples_res, (0, 2, 1, 3, 4, 5)) # (bc)
    triples_res -= np.transpose(triples_res, (2, 1, 0, 3, 4, 5)) + np.transpose(triples_res, (1, 0, 2, 3, 4, 5)) # (a/bc)
    return triples_res

def quadruples_residual(t1, t2, t3, t4, f, g, o, v):
    """Compute the projection of the CCSDTQ Hamiltonian on quadruples
        X[a, b, c, d, i, j, k, l] = < ijklabcd | (H_N exp(T1+T2+T3+T4))_C | 0 >
    """

    H1, H2 = get_ccsd_intermediates(t1, t2, f, g, o, v)

    H2[v, o, o, o] += 0.5 * np.einsum("mnef,aefijn->amij", g[o, o, v, v], t3, optimize=True)
    H2[v, v, o, v] -= 0.5 * np.einsum("mnef,abfimn->abie", g[o, o, v, v], t3, optimize=True)

    # <ijklabcd | H(2) | 0 >
    quadruples_residual = -(144.0 / 576.0) * np.einsum("amie,bcmk,edjl->abcdijkl", H2[v, o, o, v], t2, t2, optimize=True)  # (jl/i/k)(bc/a/d) = 12 * 12 = 144
    quadruples_residual += (36.0 / 576.0) * np.einsum("mnij,adml,bcnk->abcdijkl", H2[o, o, o, o], t2, t2, optimize=True)   # (ij/kl)(bc/ad) = 6 * 6 = 36
    quadruples_residual += (36.0 / 576.0) * np.einsum("abef,fcjk,edil->abcdijkl", H2[v, v, v, v], t2, t2, optimize=True)   # (jk/il)(ab/cd) = 6 * 6 = 36

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    quadruples_residual += (24.0 / 576.0) * np.einsum("cdke,abeijl->abcdijkl", H2[v, v, o, v], t3, optimize=True) # (cd/ab)(k/ijl) = 6 * 4 = 24
    quadruples_residual -= (24.0 / 576.0) * np.einsum("cmkl,abdijm->abcdijkl", H2[v, o, o, o], t3, optimize=True) # (c/abd)(kl/ij) = 6 * 4 = 24

    I_vooooo = np.einsum("nmle,bejk->bmnjkl", H2[o, o, o, v], t2, optimize=True)
    I_vooooo -= np.transpose(I_vooooo, (0, 1, 2, 5, 4, 3)) + np.transpose(I_vooooo, (0, 1, 2, 3, 5, 4))
    I_vooooo += 0.5 * np.einsum("mnef,befjkl->bmnjkl", g[o, o, v, v], t3, optimize=True) 
    quadruples_residual += 0.5 * (16.0 / 576.0) * np.einsum("bmnjkl,acdimn->abcdijkl", I_vooooo, t3, optimize=True) # (b/acd)(i/jkl) = 4 * 4 = 16

    I_vvvovv = -np.einsum("dmfe,bcjm->bcdjef", H2[v, o, v, v], t2, optimize=True)
    I_vvvovv -= np.transpose(I_vvvovv, (2, 1, 0, 3, 4, 5)) + np.transpose(I_vvvovv, (0, 2, 1, 3, 4, 5))
    quadruples_residual += 0.5 * (16.0 / 576.0) * np.einsum("bcdjef,aefikl->abcdijkl", I_vvvovv, t3, optimize=True) # (a/bcd)(j/ikl) = 4 * 4 = 16

    I_vvooov = (
                    -0.5 * np.einsum("nmke,cdnl->cdmkle", H2[o, o, o, v], t2, optimize=True)
                    +0.5 * np.einsum("cmfe,fdkl->cdmkle", H2[v, o, v, v], t2, optimize=True)
                    +0.125 * np.einsum("mnef,cdfkln->cdmkle", g[o, o, v, v], t3, optimize=True) # (ij/kl)(c/ab), compensate by factor of 1/2 !!!
    )
    I_vvooov -= np.transpose(I_vvooov, (0, 1, 2, 4, 3, 5))
    I_vvooov -= np.transpose(I_vvooov, (1, 0, 2, 3, 4, 5))
    quadruples_residual += (36.0 / 576.0) * np.einsum("cdmkle,abeijm->abcdijkl", I_vvooov, t3, optimize=True) # (cd/ab)(kl/ij) = 6 * 6 = 36

    # <ijklabcd | (H(2)*T4)_C | 0 >
    quadruples_residual -= (4.0 / 576.0) * np.einsum("mi,abcdmjkl->abcdijkl", H1[o, o], t4, optimize=True) # (l/ijk) = 4
    quadruples_residual += (4.0 / 576.0) * np.einsum("ae,ebcdijkl->abcdijkl", H1[v, v], t4, optimize=True) # (d/abc) = 4
    quadruples_residual += (6.0 / 576.0) * 0.5 * np.einsum("mnij,abcdmnkl->abcdijkl", H2[o, o, o, o], t4, optimize=True) # (kl/ij) = 6
    quadruples_residual += (6.0 / 576.0) * 0.5 * np.einsum("abef,efcdijkl->abcdijkl", H2[v, v, v, v], t4, optimize=True) # (cd/ab) = 6
    quadruples_residual += (16.0 / 576.0) * np.einsum("amie,ebcdmjkl->abcdijkl", H2[v, o, o, v], t4, optimize=True) # (d/abc)(l/ijk) = 16

    I_vvvoov = -0.5 * np.einsum("mnef,bcdfjkmn->bcdjke", g[o, o, v, v], t4, optimize=True)
    quadruples_residual += (24.0 / 576.0) * np.einsum("bcdjke,aeil->abcdijkl", I_vvvoov, t2, optimize=True) # (a/bcd)(jk/il) = 4 * 6 = 24

    I_vvoooo = 0.5 * np.einsum("mnef,bcefjkln->bcmjkl", g[o, o, v, v], t4, optimize=True)
    quadruples_residual -= (24.0 / 576.0) * np.einsum("bcmjkl,adim->abcdijkl", I_vvoooo, t2, optimize=True) # (bc/ad)(i/jkl) = 6 * 4 = 24

    quadruples_residual -= np.transpose(quadruples_residual, (0, 1, 2, 3, 4, 6, 5, 7)) # (jk)
    quadruples_residual -= np.transpose(quadruples_residual, (0, 1, 2, 3, 4, 7, 6, 5)) + np.transpose(quadruples_residual, (0, 1, 2, 3, 4, 5, 7, 6)) # (l/jk)
    quadruples_residual -= np.transpose(quadruples_residual, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(quadruples_residual, (0, 1, 2, 3, 6, 5, 4, 7)) + np.transpose(quadruples_residual, (0, 1, 2, 3, 7, 5, 6, 4)) # (i/jkl)

    quadruples_residual -= np.transpose(quadruples_residual, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    quadruples_residual -= np.transpose(quadruples_residual, (0, 3, 2, 1, 4, 5, 6, 7)) + np.transpose(quadruples_residual, (0, 1, 3, 2, 4, 5, 6, 7)) # (d/bc)
    quadruples_residual -= np.transpose(quadruples_residual, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(quadruples_residual, (2, 1, 0, 3, 4, 5, 6, 7)) + np.transpose(quadruples_residual, (3, 1, 2, 0, 4, 5, 6, 7)) # (a/bcd)
    return quadruples_residual

def kernel(fock, g, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core, use_quasi):
    """Solve the CCSDTQ system of nonlinear equations using Jacobi iterations
    with DIIS acceleration. The initial values of the T amplitudes are taken to be 0."""

    eps = np.diagonal(fock)
    n = np.newaxis
    e_abcdijkl = 1.0 / (- eps[v, n, n, n, n, n, n, n] - eps[n, v, n, n, n, n, n, n] - eps[n, n, v, n, n, n, n, n] - eps[n, n, n, v, n, n, n, n]
                      + eps[n, n, n, n, o, n, n, n] + eps[n, n, n, n, n, o, n, n] + eps[n, n, n, n, n, n, o, n] + eps[n, n, n, n, n, n, n, o] + energy_shift )
    e_abcijk = 1.0 / (- eps[v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                    + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o] + energy_shift )
    e_abij = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o] + energy_shift )
    e_ai = 1.0 / (-eps[v, n] + eps[n, o] + energy_shift )

    nunocc, nocc = e_ai.shape
    n1 = nocc * nunocc
    n2 = nocc**2 * nunocc**2
    n3 = nocc**3 * nunocc**3
    n4 = nocc**4 * nunocc**4
    ndim = n1 + n2 + n3 + n4

    diis_engine = DIIS(ndim, diis_size, out_of_core)

    t1 = np.zeros((nunocc, nocc))
    t2 = np.zeros((nunocc, nunocc, nocc, nocc))
    t3 = np.zeros((nunocc, nunocc, nunocc, nocc, nocc, nocc))
    t4 = np.zeros((nunocc, nunocc, nunocc, nunocc, nocc, nocc, nocc, nocc))
    old_energy = cc_energy(t1, t2, fock, g, o, v)

    print("    ==> CCSDTQ amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|     Wall Time     Memory")
    for idx in range(maxit):

        tic = time.time()

        residual_singles = singles_residual(t1, t2, t3, fock, g, o, v)
        residual_doubles = doubles_residual(t1, t2, t3, t4, fock, g, o, v)
        residual_triples = triples_residual(t1, t2, t3, t4, fock, g, o, v)
        residual_quadruples = quadruples_residual(t1, t2, t3, t4, fock, g, o, v)

        res_norm = np.linalg.norm(residual_singles) + np.linalg.norm(residual_doubles) + np.linalg.norm(residual_triples) + np.linalg.norm(residual_quadruples)

        t1 += residual_singles * e_ai
        t2 += residual_doubles * e_abij
        t3 += residual_triples * e_abcijk
        t4 += residual_quadruples * e_abcdijkl

        current_energy = cc_energy(t1, t2, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < convergence and res_norm < convergence:
            break

        if idx >= n_start_diis:
            diis_engine.push( (t1, t2, t3, t4), (residual_singles, residual_doubles, residual_triples, residual_quadruples), idx) 

        if idx >= diis_size + n_start_diis:
            T_extrap = diis_engine.extrapolate()
            t1 = T_extrap[:n1].reshape((nunocc, nocc))
            t2 = T_extrap[n1:n1+n2].reshape((nunocc, nunocc, nocc, nocc))
            t3 = T_extrap[n1+n2:n1+n2+n3].reshape((nunocc, nunocc, nunocc, nocc, nocc, nocc))
            t4 = T_extrap[n1+n2+n3:].reshape((nunocc, nunocc, nunocc, nunocc, nocc, nocc, nocc, nocc))

        old_energy = current_energy

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(idx, current_energy, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        raise ValueError("CCSDTQ iterations did not converge")

    diis_engine.cleanup()
    
    e_corr = cc_energy(t1, t2, fock, g, o, v)

    return (t1, t2, t3, t4), e_corr


