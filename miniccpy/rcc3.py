import time
import numpy as np
from miniccpy.energy import rcc_energy
from miniccpy.helper_cc import get_rccs_intermediates
from miniccpy.diis import DIIS
from miniccpy.utilities import get_memory_usage

def singles_residual(t1, t2, f, g, o, v):
    """Compute the projection of the CCSDT Hamiltonian on singles
        X[a, i] = < ia | (H_N exp(T1+T2+T3))_C | 0 >
    """
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
    return singles_res

def doubles_residual(t1, t2, f, g, o, v):
    """Compute the projection of the CCSDT Hamiltonian on doubles
        X[a, b, i, j] = < ijab | (H_N exp(T1+T2+T3))_C | 0 >
    """
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

def compute_ccs_intermediates(f, g, t1, t2, o, v):
    Q1 = -np.einsum("nmef,an->amef", g[o, o, v, v], t1, optimize=True)
    I_vovv = g[v, o, v, v] + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    I_ooov = g[o, o, o, v] + 0.5 * Q1

    I_vvvv = g[v, v, v, v] + (
            - np.einsum("amef,bm->abef", I_vovv, t1, optimize=True)
            - np.einsum("bmfe,am->abef", I_vovv, t1, optimize=True)
    )

    I_oooo = g[o, o, o, o] + (
              np.einsum("mnie,ej->mnij", I_ooov, t1, optimize=True)
            + np.einsum("nmje,ei->mnij", I_ooov, t1, optimize=True)
    )

    Q1 = g[v, o, o, v] + np.einsum("amfe,fi->amie", g[v, o, v, v], t1, optimize=True)
    I_vooo = g[v, o, o, o] + (
            - np.einsum("nmij,an->amij", I_oooo, t1, optimize=True)
            + np.einsum("amej,ei->amij", g[v, o, v, o], t1, optimize=True)
            + np.einsum("amie,ej->amij", Q1, t1, optimize=True)
    )
    Q1 = g[o, v, o, v] - np.einsum("mnie,bn->mbie", g[o, o, o, v], t1, optimize=True)
    Q1 = -np.einsum("mbie,am->abie", Q1, t1, optimize=True)
    I_vvov = g[v, v, o, v] + Q1 + (
            + np.einsum("abfe,fi->abie", I_vvvv, t1, optimize=True)
            - np.einsum("amie,bm->abie", g[v, o, o, v], t1, optimize=True)
    )
    return I_vooo, I_vvov

def add_t3_contributions(t1, t2, f, g, I_vooo, I_vvov, e_abc, o, v):
    nu, no = t1.shape
    # Intermediates
    h_ov = f[o, v] + (
              np.einsum("mnef,fn->me", g[o, o, v, v], t1, optimize=True)
            - np.einsum("mnfe,fn->me", g[o, o, v, v], t1, optimize=True)
            + np.einsum("mnef,fn->me", g[o, o, v, v], t1, optimize=True)
    )
    h_ooov = g[o, o, o, v] + np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    h_vovv = g[v, o, v, v] - np.einsum("nmef,an->amef", g[o, o, v, v], t1, optimize=True)
    # RHF-adapted integral element
    gs_oovv = 2.0 * g[o, o, v, v] - g[o, o, v, v].swapaxes(2, 3)
    # residual containers
    singles_res = np.zeros((nu, no))
    doubles_res = np.zeros((nu, nu, no, no))
    for i in range(no):
        for j in range(no):
            for k in range(no):
                if i == j and j == k: continue
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                # -h2(amij) * t2(bcmk)
                m3 = -np.einsum("am,bcm->abc", I_vooo[:, :, i, j], t2[:, :, :, k], optimize=True) # (1)
                m3 -= np.einsum("bm,acm->abc", I_vooo[:, :, j, i], t2[:, :, :, k], optimize=True) # (ij)(ab)
                m3 -= np.einsum("cm,bam->abc", I_vooo[:, :, k, j], t2[:, :, :, i], optimize=True) # (ac)(ik)
                m3 -= np.einsum("am,cbm->abc", I_vooo[:, :, i, k], t2[:, :, :, j], optimize=True) # (bc)(jk)
                m3 -= np.einsum("bm,cam->abc", I_vooo[:, :, j, k], t2[:, :, :, i], optimize=True) # (ij)(ab)(ac)(ik)
                m3 -= np.einsum("cm,abm->abc", I_vooo[:, :, k, i], t2[:, :, :, j], optimize=True) # (ij)(ab)(bc)(jk)
                # h2(abie) * t2(bcek)
                m3 += np.einsum("abe,ec->abc", I_vvov[:, :, i, :], t2[:, :, j, k], optimize=True) # (1)
                m3 += np.einsum("bae,ec->abc", I_vvov[:, :, j, :], t2[:, :, i, k], optimize=True) # (ij)(ab)
                m3 += np.einsum("cbe,ea->abc", I_vvov[:, :, k, :], t2[:, :, j, i], optimize=True) # (ac)(ik)
                m3 += np.einsum("ace,eb->abc", I_vvov[:, :, i, :], t2[:, :, k, j], optimize=True) # (bc)(jk)
                m3 += np.einsum("bce,ea->abc", I_vvov[:, :, j, :], t2[:, :, k, i], optimize=True) # (ij)(ab)(ac)(ik)
                m3 += np.einsum("cae,eb->abc", I_vvov[:, :, k, :], t2[:, :, i, j], optimize=True) # (ij)(ab)(bc)(jk)
                # divide by MP denominator
                m3 /= (e_abc + denom_occ)
                # zero out diagonal elements
                for a in range(nu):
                    m3[a, a, a] *= 0.0
                # update singles residual
                singles_res[:, i] += np.einsum('abc,bc->a', m3 - m3.swapaxes(0, 2), gs_oovv[j, k, :, :], optimize=True)
                # symmetrize
                m3 = (2.0 * m3
                      - m3.swapaxes(1, 2)
                      - m3.swapaxes(0, 2)
                )
                # update doubles residual
                doubles_res[:, :, i, j] += 0.5 * np.einsum('abc,c->ab', m3, h_ov[k, :])
                doubles_res[:, :, i, j] += np.einsum('abc,dbc->ad', m3, h_vovv[:, k, :, :])
                doubles_res[:, :, i, :] -= np.einsum('abc,lc->abl', m3, h_ooov[j, k, :, :])
    # Apply (ij)(ab) symmetrizer
    doubles_res += doubles_res.transpose(1, 0, 3, 2)
    return singles_res, doubles_res

def compute_t3(I_vooo, I_vvov, t2, e_abcijk, fock, e_abc):
    """Compute the projection of the CCSDT Hamiltonian on triples
        X[a, b, c, i, j, k] = < ijkabc | (H_N exp(T1+T2+T3))_C | 0 >
    """
    # -h2(amij) * t2(bcmk)
    triples_res = -np.einsum("amij,bcmk->abcijk", I_vooo, t2, optimize=True)
    # h2(abie) * t2(bcek)
    triples_res += np.einsum("abie,ecjk->abcijk", I_vvov, t2, optimize=True)
    # [1 + P(ai/bj)][1 + P(ai/ck) + P(bj/ck)] = 1 + P(ai/bj) + P(ai/ck) + P(bj/ck) + P(ai/bj)P(ai/ck) + P(ai/bj)P(bj/ck)
    triples_res += (    triples_res.transpose(1, 0, 2, 4, 3, 5)   # (ij)(ab)
                      + triples_res.transpose(2, 1, 0, 5, 4, 3)   # (ac)(ik)
                      + triples_res.transpose(0, 2, 1, 3, 5, 4)   # (bc)(jk)
                      + triples_res.transpose(2, 0, 1, 5, 3, 4)   # (ab)(ij)(ac)(ik)
                      + triples_res.transpose(1, 2, 0, 4, 5, 3) ) # (ab)(ij)(bc)(jk)
    # Manually zero out the i = j = k and a = b = c blocks
    nu, _, no, _ = t2.shape
    for i in range(no):
        triples_res[:, :, :, i, i, i] *= 0.0
    for a in range(nu):
        triples_res[a, a, a, :, :, :] *= 0.0

    # error = 0.0
    # for i in range(no):
    #     for j in range(i, no):
    #         for k in range(j, no):
    #             if i == j and j == k: continue
    #             # -h2(amij) * t2(bcmk)
    #             m3 = -np.einsum("am,bcm->abc", I_vooo[:, :, i, j], t2[:, :, :, k], optimize=True) # (1)
    #             m3 -= np.einsum("bm,acm->abc", I_vooo[:, :, j, i], t2[:, :, :, k], optimize=True) # (ij)(ab)
    #             m3 -= np.einsum("cm,bam->abc", I_vooo[:, :, k, j], t2[:, :, :, i], optimize=True) # (ac)(ik)
    #             m3 -= np.einsum("am,cbm->abc", I_vooo[:, :, i, k], t2[:, :, :, j], optimize=True) # (bc)(jk)
    #             m3 -= np.einsum("bm,cam->abc", I_vooo[:, :, j, k], t2[:, :, :, i], optimize=True) # (ij)(ab)(ac)(ik)
    #             m3 -= np.einsum("cm,abm->abc", I_vooo[:, :, k, i], t2[:, :, :, j], optimize=True) # (ij)(ab)(bc)(jk)
    #             # h2(abie) * t2(bcek)
    #             m3 += np.einsum("abe,ec->abc", I_vvov[:, :, i, :], t2[:, :, j, k], optimize=True) # (1)
    #             m3 += np.einsum("bae,ec->abc", I_vvov[:, :, j, :], t2[:, :, i, k], optimize=True) # (ij)(ab)
    #             m3 += np.einsum("cbe,ea->abc", I_vvov[:, :, k, :], t2[:, :, j, i], optimize=True) # (ac)(ik)
    #             m3 += np.einsum("ace,eb->abc", I_vvov[:, :, i, :], t2[:, :, k, j], optimize=True) # (bc)(jk)
    #             m3 += np.einsum("bce,ea->abc", I_vvov[:, :, j, :], t2[:, :, k, i], optimize=True) # (ij)(ab)(ac)(ik)
    #             m3 += np.einsum("cae,eb->abc", I_vvov[:, :, k, :], t2[:, :, i, j], optimize=True) # (ij)(ab)(bc)(jk)
    #             # zero out diagonal elements
    #             for a in range(nu):
    #                 m3[a, a, a] *= 0.0
    #             error += np.linalg.norm(m3.flatten() - triples_res[:, :, :, i, j, k].flatten())
    # print(error)
    return triples_res * e_abcijk

def kernel(fock, g, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core, use_quasi):
    """Solve the CCSDT system of nonlinear equations using Jacobi iterations
    with DIIS acceleration. The initial values of the T amplitudes are taken to be 0."""

    eps = np.diagonal(fock)
    n = np.newaxis
    e_abc = -eps[v, n, n] - eps[n, v, n] - eps[n, n, v]
    e_abij = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o] + energy_shift )
    e_ai = 1.0 / (-eps[v, n] + eps[n, o] + energy_shift )

    nunocc, nocc = e_ai.shape
    n1 = nocc * nunocc
    n2 = nocc**2 * nunocc**2
    ndim = n1 + n2

    diis_engine = DIIS(ndim, diis_size, out_of_core)

    t1 = np.zeros((nunocc, nocc))
    t2 = np.zeros((nunocc, nunocc, nocc, nocc))

    old_energy = rcc_energy(t1, t2, fock, g, o, v)

    print("    ==> R-CC3 amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|     Wall Time     Memory")
    for idx in range(maxit):

        tic = time.time()

        # Compute T3 using the perturbative approximation of CC3
        I_vooo, I_vvov = compute_ccs_intermediates(fock, g, t1, t2, o, v)
        residual_singles, residual_doubles = add_t3_contributions(t1, t2, fock, g, I_vooo, I_vvov, e_abc, o, v)
        residual_singles += singles_residual(t1, t2, fock, g, o, v)
        residual_doubles += doubles_residual(t1, t2, fock, g, o, v)

        res_norm = np.linalg.norm(residual_singles) + np.linalg.norm(residual_doubles)

        t1 += residual_singles * e_ai
        t2 += residual_doubles * e_abij

        current_energy = rcc_energy(t1, t2, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < convergence and res_norm < convergence:
            break
 
        if idx >= n_start_diis:
            diis_engine.push( (t1, t2), (residual_singles, residual_doubles), idx) 

        if idx >= diis_size + n_start_diis:
            T_extrap = diis_engine.extrapolate()
            t1 = T_extrap[:n1].reshape((nunocc, nocc))
            t2 = T_extrap[n1:].reshape((nunocc, nunocc, nocc, nocc))

        old_energy = current_energy

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(idx, current_energy, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        raise ValueError("CC3 iterations did not converge")

    diis_engine.cleanup()
    e_corr = rcc_energy(t1, t2, fock, g, o, v)

    return (t1, t2), e_corr


