import time
import numpy as np
from miniccpy.energy import rcc_energy
from miniccpy.hbar import get_rccs_intermediates, get_rccsd_intermediates
from miniccpy.diis import DIIS

def singles_residual(t1, t2, t3, f, g, o, v):
    """Compute the projection of the CCSDT Hamiltonian on singles
        X[a, i] = < ia | (H_N exp(T1+T2+T3))_C | 0 >
    """
    # spin-summed t3: t3(ABcIJk) = 2*t3(AbcIjk) - t3(Abc
    #                            = 2*(2*t3(abcijk) - t3(abcjik) - t3(abckji)) - 
    t3_ss = (
                4.0 * t3 
                - 2.0 * t3.transpose(0, 1, 2, 4, 3, 5) 
                - 2.0 * t3.transpose(0, 1, 2, 5, 4, 3) 
                - 2.0 * t3.transpose(0, 1, 2, 3, 5, 4) 
                + t3.transpose(0, 1, 2, 4, 5, 3) 
                + t3.transpose(0, 1, 2, 5, 3, 4)
    )
    # intermediates
    I_ov = (
             f[o, v]
           + 2.0 * np.einsum("mnef,fn->me", g[o, o, v, v], t1, optimize=True)
           - np.einsum("mnfe,fn->me", g[o, o, v, v], t1, optimize=True)
    )
    I_vv = (
            f[v, v]
            + 2.0 * np.einsum("anef,fn->ae", g[v, o, v, v], t1, optimize=True)
            - np.einsum("anfe,fn->ae", g[v, o, v, v], t1, optimize=True)
    )
    I_oo = (
            f[o, o]
            + 2.0 * np.einsum("mnif,fn->mi", g[o, o, o, v], t1, optimize=True)
            - np.einsum("nmif,fn->mi", g[o, o, o, v], t1, optimize=True)
            + np.einsum("me,ei->mi", I_ov, t1, optimize=True)
    )
    I_ooov = g[o, o, o, v] + np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    I_vovv = g[v, o, v, v] - np.einsum("nmef,an->amef", g[o, o, v, v], t1, optimize=True)

    singles_res = -np.einsum("mi,am->ai", I_oo, t1, optimize=True)
    singles_res += np.einsum("ae,ei->ai", I_vv, t1, optimize=True)
    singles_res += 2.0 * np.einsum("me,aeim->ai", I_ov, t2, optimize=True)
    singles_res -= np.einsum("me,aemi->ai", I_ov, t2, optimize=True)
    singles_res += 2.0 * np.einsum("anif,fn->ai", g[v, o, o, v], t1, optimize=True)
    singles_res -= np.einsum("anfi,fn->ai", g[v, o, v, o], t1, optimize=True)
    singles_res -= 2.0 * np.einsum("mnif,afmn->ai", I_ooov, t2, optimize=True)
    singles_res += np.einsum("nmif,afmn->ai", I_ooov, t2, optimize=True)
    singles_res += 2.0 * np.einsum("anef,efin->ai", I_vovv, t2, optimize=True)
    singles_res -= np.einsum("anfe,efin->ai", I_vovv, t2, optimize=True)
    singles_res += f[v, o]
    # T3 parts
    singles_res += 0.5 * np.einsum("mnef,aefimn->ai", g[o, o, v, v], t3_ss, optimize=True)
    return singles_res

def doubles_residual(t1, t2, t3, f, g, o, v):
    """Compute the projection of the CCSDT Hamiltonian on doubles
        X[a, b, i, j] = < ijab | (H_N exp(T1+T2+T3))_C | 0 >
    """
    # partially spin-summed t3(AbcIjk) = 2*t3(abcijk) - t3(abcjik) - t3(abckji)
    t3_s = (
            2.0 * t3
            - t3.transpose(0, 1, 2, 4, 3, 5)
            - t3.transpose(0, 1, 2, 5, 4, 3)
    )
    # intermediates
    H1, H2 = get_rccs_intermediates(t1, f, g, o, v)
    I_vv = (
        H1[v, v]
        - 2.0 * np.einsum("mnef,afmn->ae", g[o, o, v, v], t2, optimize=True)
        + np.einsum("mnef,afnm->ae", g[o, o, v, v], t2, optimize=True)
    )
    I_oo = (
        H1[o, o]
        + 2.0 * np.einsum("mnef,efin->mi", g[o, o, v, v], t2, optimize=True)
        - np.einsum("mnef,efni->mi", g[o, o, v, v], t2, optimize=True)
    )
    I_voov = (
        H2[v, o, o, v]
        + 2.0 * np.einsum("mnef,aeim->anif", g[o, o, v, v], t2, optimize=True)
        - np.einsum("mnef,aemi->anif", g[o, o, v, v], t2, optimize=True)
        - np.einsum("mnfe,aeim->anif", g[o, o, v, v], t2, optimize=True)
    )
    I_oooo = H2[o, o, o, o] + np.einsum("mnef,efij->mnij", g[o, o, v, v], t2, optimize=True)
    I_vovo = H2[v, o, v, o] - np.einsum("mnef,afmj->anej", g[o, o, v, v], t2, optimize=True)
    I_ovoo = H2[o, v, o, o] + np.einsum("amfe,efij->maij", g[v, o, v, v] + 0.5 * H2[v, o, v, v], t2, optimize=True)
    I_vooo = H2[v, o, o, o] + np.einsum("amef,efij->amij", g[v, o, v, v] + 0.5 * H2[v, o, v, v], t2, optimize=True)
    tau = t2 + np.einsum('ai,bj->abij', t1, t1, optimize=True)
    # collect terms that can be symmetrized
    doubles_res = np.einsum("baje,ei->abij", H2[v, v, o, v], t1, optimize=True)
    doubles_res += np.einsum("ae,ebij->abij", I_vv, t2, optimize=True)
    doubles_res -= np.einsum("mi,abmj->abij", I_oo, t2, optimize=True)
    doubles_res += 0.5 * np.einsum("mnij,abmn->abij", I_oooo, t2, optimize=True)
    doubles_res += 0.5 * np.einsum("abef,efij->abij", g[v, v, v, v], tau, optimize=True)
    doubles_res += 0.5 * g[v, v, o, o]
    # T3 parts
    doubles_res += 0.5 * np.einsum("me,eabmij->abij", H1[o, v], t3_s, optimize=True)
    doubles_res += np.einsum("amef,febmij->abij", g[v, o, v, v] + H2[v, o, v, v], t3_s, optimize=True)
    doubles_res -= np.einsum("nmje,eabmin->abij", g[o, o, o, v] + H2[o, o, o, v], t3_s, optimize=True)
    doubles_res += doubles_res.transpose(1, 0, 3, 2)
    # remaining terms
    # can be made into (ij)(ab) pairs
    doubles_res -= np.einsum("mbij,am->abij", I_ovoo, t1, optimize=True)
    doubles_res -= np.einsum("amij,bm->abij", I_vooo, t1, optimize=True)
    # can be made into (ij)(ab) pairs
    doubles_res += 2.0 * np.einsum("amie,ebmj->abij", I_voov, t2, optimize=True)
    doubles_res -= np.einsum("amie,ebjm->abij", I_voov, t2, optimize=True)
    doubles_res += 2.0 * np.einsum("bmje,aeim->abij", H2[v, o, o, v], t2, optimize=True)
    doubles_res -= np.einsum("bmje,aemi->abij", H2[v, o, o, v], t2, optimize=True)
    # can be made into (ij)(ab) pairs
    doubles_res -= np.einsum("bmej,aeim->abij", H2[v, o, v, o], t2, optimize=True)
    doubles_res -= np.einsum("amei,bejm->abij", I_vovo, t2, optimize=True)
    # can be made into (ij)(ab) pairs
    doubles_res -= np.einsum("bmei,aemj->abij", H2[v, o, v, o], t2, optimize=True)
    doubles_res -= np.einsum("amej,ebim->abij", I_vovo, t2, optimize=True)
    return doubles_res

def triples_residual_110111(t1, t2, t3, f, g, o, v):
    """Compute the projection of the CCSDT Hamiltonian on triples
        X[a, b, c, i, j, k] = < ijkabc | (H_N exp(T1+T2+T3))_C | 0 >
    """
    # partially spin-summed t3(AbcIjk) = 2*t3(abcijk) - t3(abcjik) - t3(abckji)
    t3_s = (
            2.0 * t3
            - t3.transpose(0, 1, 2, 4, 3, 5)
            - t3.transpose(0, 1, 2, 5, 4, 3)
    )
    # intermediates
    H1, H2 = get_rccsd_intermediates(t1, t2, f, g, o, v)
    I_vvov = H2[v, v, o, v] - np.einsum("nmfe,fabnim->abie", g[o, o, v, v], t3_s, optimize=True)
    I_vooo = H2[v, o, o, o] + (
            - np.einsum("me,aeik->amik", H1[o, v], t2, optimize=True)
            + np.einsum("nmfe,faenij->amij", g[o, o, v, v], t3_s, optimize=True)
    )

    # MM(2,3)B
    triples_res = -np.einsum("amij,bcmk->abcijk", I_vooo, t2, optimize=True)
    triples_res += np.einsum("abie,ecjk->abcijk", I_vvov, t2, optimize=True)
    # (HBar*T3)_C
    triples_res += 0.5 * np.einsum("AE,EBcIJK->ABcIJK", H1[v, v], t3, optimize=True)
    triples_res += 0.5 * np.einsum("Ae,BecJIK->ABcIJK", H1[v, v], t3, optimize=True)


    triples_res -= 0.5 * np.einsum("mi,abcmjk->abcijk", H1[o, o], t3, optimize=True)
    triples_res += 0.5 * np.einsum("mnij,abcmnk->abcijk", H2[o, o, o, o], t3, optimize=True)
    triples_res += 0.5 * np.einsum("abef,efcijk->abcijk", H2[v, v, v, v], t3, optimize=True)
    triples_res += 0.5 * np.einsum("amie,ebcmjk->abcijk", H2[v, o, o, v], t3_s, optimize=True)
    triples_res -= 0.25 * np.einsum("amei,ebcmjk->abcijk", H2[v, o, v, o], t3_s, optimize=True)
    triples_res -= 0.5 * np.einsum("amei,ebcjmk->abcijk", H2[v, o, v, o], t3, optimize=True)
    triples_res -= np.einsum("bmei,eacjmk->abcijk", H2[v, o, v, o], t3, optimize=True)
    # [1 + P(ai/bj)][1 + P(ai/ck) + P(bj/ck)] = 1 + P(ai/bj) + P(ai/ck) + P(bj/ck) + P(ai/bj)P(ai/ck) + P(ai/bj)P(bj/ck)
    triples_res += (    triples_res.transpose(1, 0, 2, 4, 3, 5)   # (ij)(ab)
                      + triples_res.transpose(2, 1, 0, 5, 4, 3)   # (ac)(ik)
                      + triples_res.transpose(0, 2, 1, 3, 5, 4)   # (bc)(jk)
                      + triples_res.transpose(2, 0, 1, 5, 3, 4)   # (ab)(ij)(ac)(ik)
                      + triples_res.transpose(1, 2, 0, 4, 5, 3) ) # (ab)(ij)(bc)(jk)
    return triples_res


def kernel(fock, g, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core, use_quasi):
    """Solve the CCSDT system of nonlinear equations using Jacobi iterations
    with DIIS acceleration. The initial values of the T amplitudes are taken to be 0."""

    eps = np.diagonal(fock)
    n = np.newaxis
    e_abcijk = 1.0 / (- eps[v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                    + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o] + energy_shift )
    e_abij = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o] + energy_shift )
    e_ai = 1.0 / (-eps[v, n] + eps[n, o] + energy_shift )

    nunocc, nocc = e_ai.shape
    n1 = nocc * nunocc
    n2 = nocc**2 * nunocc**2
    n3 = nocc**3 * nunocc**3
    ndim = n1 + n2 + n3

    diis_engine = DIIS(ndim, diis_size, out_of_core)

    t1 = np.zeros((nunocc, nocc))
    t2 = np.zeros((nunocc, nunocc, nocc, nocc))
    t3 = np.zeros((nunocc, nunocc, nunocc, nocc, nocc, nocc))

    old_energy = rcc_energy(t1, t2, fock, g, o, v)

    print("    ==> R-CCSDT amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(maxit):

        tic = time.time()

        residual_singles = singles_residual(t1, t2, t3, fock, g, o, v)
        residual_doubles = doubles_residual(t1, t2, t3, fock, g, o, v)
        residual_triples = triples_residual(t1, t2, t3, fock, g, o, v)

        res_norm = np.linalg.norm(residual_singles) + np.linalg.norm(residual_doubles) + np.linalg.norm(residual_triples)

        t1 += residual_singles * e_ai
        t2 += residual_doubles * e_abij
        t3 += residual_triples * e_abcijk

        current_energy = rcc_energy(t1, t2, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < convergence and res_norm < convergence:
            break
 
        if idx >= n_start_diis:
            diis_engine.push( (t1, t2, t3), (residual_singles, residual_doubles, residual_triples), idx) 

        if idx >= diis_size + n_start_diis:
            T_extrap = diis_engine.extrapolate()
            t1 = T_extrap[:n1].reshape((nunocc, nocc))
            t2 = T_extrap[n1:n1+n2].reshape((nunocc, nunocc, nocc, nocc))
            t3 = T_extrap[n1+n2:].reshape((nunocc, nunocc, nunocc, nocc, nocc, nocc))

        old_energy = current_energy

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(idx, current_energy, delta_e, res_norm, minutes, seconds))
    else:
        raise ValueError("CCSDT iterations did not converge")

    diis_engine.cleanup()
    e_corr = rcc_energy(t1, t2, fock, g, o, v)

    return (t1, t2, t3), e_corr


