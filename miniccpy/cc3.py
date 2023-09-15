import time
import numpy as np
from miniccpy.energy import cc_energy, hf_energy, hf_energy_from_fock
from miniccpy.hbar import get_ccs_intermediates, get_ccsd_intermediates
from miniccpy.diis import DIIS

def singles_residual(t1, t2, t3, f, g, o, v, I_vooo, I_vvov, e_abc):
    """Compute the projection of the CCSDT Hamiltonian on singles
        X[a, i] = < ia | (H_N exp(T1+T2+T3))_C | 0 >
    """

    no, nu = f[o, v].shape

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

    #for i in range(no):
    #    for j in range(i + 1, no):
    #        for k in range(j + 1, no):
    #            # fock denominator for occupied
    #            denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
    #            # -1/2 A(k/ij)A(abc) I(amij) * t(bcmk)
    #            t3_abc = -0.5 * np.einsum("am,bcm->abc", I_vooo[:, :, i, j], t2[:, :, :, k], optimize=True)
    #            t3_abc += 0.5 * np.einsum("am,bcm->abc", I_vooo[:, :, k, j], t2[:, :, :, i], optimize=True)
    #            t3_abc += 0.5 * np.einsum("am,bcm->abc", I_vooo[:, :, i, k], t2[:, :, :, j], optimize=True)
    #            # 1/2 A(i/jk)A(abc) I(abie) * t(ecjk)
    #            t3_abc += 0.5 * np.einsum("abe,ec->abc", I_vvov[:, :, i, :], t2[:, :, j, k], optimize=True)
    #            t3_abc -= 0.5 * np.einsum("abe,ec->abc", I_vvov[:, :, j, :], t2[:, :, i, k], optimize=True)
    #            t3_abc -= 0.5 * np.einsum("abe,ec->abc", I_vvov[:, :, k, :], t2[:, :, j, i], optimize=True)
    #            # Antisymmetrize A(abc)
    #            t3_abc -= np.transpose(t3_abc, (1, 0, 2)) + np.transpose(t3_abc, (2, 1, 0)) # A(a/bc)
    #            t3_abc -= np.transpose(t3_abc, (0, 2, 1)) # A(bc)
    #            # Divide t_abc by the denominator
    #            t3_abc /= (denom_occ + e_abc)
    #            # Compute diagram: 1/2 A(i/jk) v(jkbc) * t(abcijk)
    #            singles_res[:, i] += 0.5 * np.einsum("bc,abc->a", g[o, o, v, v][j, k, :, :], t3_abc, optimize=True)
    #            singles_res[:, j] -= 0.5 * np.einsum("bc,abc->a", g[o, o, v, v][i, k, :, :], t3_abc, optimize=True)
    #            singles_res[:, k] -= 0.5 * np.einsum("bc,abc->a", g[o, o, v, v][j, i, :, :], t3_abc, optimize=True)

    return singles_res


def doubles_residual(t1, t2, t3, f, g, o, v, I_vooo, I_vvov, e_abc):
    """Compute the projection of the CCSDT Hamiltonian on doubles
        X[a, b, i, j] = < ijab | (H_N exp(T1+T2+T3))_C | 0 >
    """

    no, nu = f[o, v].shape
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

    doubles_res -= np.transpose(doubles_res, (1, 0, 2, 3))
    doubles_res -= np.transpose(doubles_res, (0, 1, 3, 2))
    doubles_res += g[v, v, o, o]

    #doubles_res2 = np.zeros((nu, nu, no, no))
    #I_ooov = g[o, o, o, v] + H2[o, o, o, v]
    #I_vovv = g[v, o, v, v] + H2[v, o, v, v]
    #for i in range(no):
    #    for j in range(i + 1, no):
    #        for k in range(j + 1, no):
    #            # fock denominator for occupied
    #            denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
    #            # -1/2 A(k/ij)A(abc) I(amij) * t(bcmk)
    #            t3_abc = -0.5 * np.einsum("am,bcm->abc", I_vooo[:, :, i, j], t2[:, :, :, k], optimize=True)
    #            t3_abc += 0.5 * np.einsum("am,bcm->abc", I_vooo[:, :, k, j], t2[:, :, :, i], optimize=True)
    #            t3_abc += 0.5 * np.einsum("am,bcm->abc", I_vooo[:, :, i, k], t2[:, :, :, j], optimize=True)
    #            # 1/2 A(i/jk)A(abc) I(abie) * t(ecjk)
    #            t3_abc += 0.5 * np.einsum("abe,ec->abc", I_vvov[:, :, i, :], t2[:, :, j, k], optimize=True)
    #            t3_abc -= 0.5 * np.einsum("abe,ec->abc", I_vvov[:, :, j, :], t2[:, :, i, k], optimize=True)
    #            t3_abc -= 0.5 * np.einsum("abe,ec->abc", I_vvov[:, :, k, :], t2[:, :, j, i], optimize=True)
    #            # Antisymmetrize A(abc)
    #            t3_abc -= np.transpose(t3_abc, (1, 0, 2)) + np.transpose(t3_abc, (2, 1, 0)) # A(a/bc)
    #            t3_abc -= np.transpose(t3_abc, (0, 2, 1)) # A(bc)
    #            # Divide t_abc by the denominator
    #            t3_abc /= (denom_occ + e_abc)
    #            # Compute diagram: A(ij) [A(k/ij) h(ke) * t3(abeijk)]
    #            doubles_res2[:, :, i, j] += 0.5 * np.einsum("e,abe->ab", H1[o, v][k, :], t3_abc, optimize=True)
    #            doubles_res2[:, :, k, j] -= 0.5 * np.einsum("e,abe->ab", H1[o, v][i, :], t3_abc, optimize=True)
    #            doubles_res2[:, :, i, k] -= 0.5 * np.einsum("e,abe->ab", H1[o, v][j, :], t3_abc, optimize=True)
                # Compute diagram: -A(j/ik) h(ik:f) * t3(abfijk)
                #doubles_res2[:, :, :, j] -= np.einsum("mf,abf->abm", I_ooov[i, k, :, :], t3_abc, optimize=True) 
                #doubles_res2[:, :, :, i] += np.einsum("mf,abf->abm", I_ooov[j, k, :, :], t3_abc, optimize=True) 
                #doubles_res2[:, :, :, k] += np.einsum("mf,abf->abm", I_ooov[i, j, :, :], t3_abc, optimize=True) 
                # Compute diagram: 1/2 A(k/ij) h(akef) * t3(efbijk) 
                #doubles_res2[:, :, i, j] += 0.5 * np.einsum("aef,efb->ab", I_vovv[:, k, :, :], t3_abc, optimize=True) 
                #doubles_res2[:, :, k, j] -= 0.5 * np.einsum("aef,efb->ab", I_vovv[:, i, :, :], t3_abc, optimize=True) 
                #doubles_res2[:, :, i, k] -= 0.5 * np.einsum("aef,efb->ab", I_vovv[:, j, :, :], t3_abc, optimize=True) 


    #doubles_res2 -= np.transpose(doubles_res2, (1, 0, 2, 3))
    #doubles_res2 -= np.transpose(doubles_res2, (0, 1, 3, 2))
    ## Manually clear all diagonal elements
    #for a in range(nu):
    #    doubles_res2[a, a, :, :] *= 0.0
    #for i in range(no):
    #    doubles_res2[:, :, i, i] *= 0.0
    #return doubles_res + doubles_res2
    return doubles_res

def compute_ccs_intermediates(f, g, t1, t2, o, v):

    # h(vvov) and h(vooo) intermediates resulting from exp(-T1) H_N exp(T1)
    Q1 = -np.einsum("mnfe,an->amef", g[o, o, v, v], t1, optimize=True)
    I_vovv = g[v, o, v, v] + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    I_ooov = g[o, o, o, v] + 0.5 * Q1

    Q1 = -np.einsum("bmfe,am->abef", I_vovv, t1, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    I_vvvv = g[v, v, v, v] + Q1

    Q1 = +np.einsum("nmje,ei->mnij", I_ooov, t1, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    I_oooo = g[o, o, o, o] + Q1

    Q1 = g[v, o, o, v] + 0.5 * np.einsum("amef,ei->amif", g[v, o, v, v], t1, optimize=True)
    Q1 = np.einsum("amif,fj->amij", Q1, t1, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    I_vooo = g[v, o, o, o] + Q1 - np.einsum("nmij,an->amij", I_oooo, t1, optimize=True)
    # Added in for ROHF
    I_vooo += np.einsum("me,aeij->amij", f[o, v], t2, optimize=True)

    Q1 = g[o, v, o, v] - 0.5 * np.einsum("mnie,bn->mbie", g[o, o, o, v], t1, optimize=True)
    Q1 = -np.einsum("mbie,am->abie", Q1, t1, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    I_vvov = g[v, v, o, v] + Q1 + np.einsum("abfe,fi->abie", I_vvvv, t1, optimize=True)
    # Added in for ROHF
    #I_vvov -= np.einsum("me,abim->abie", f[o, v], t2, optimize=True)

    return I_vooo, I_vvov

def compute_t3(t1, t2, e_abcijk, f, g, o, v):
    """Compute the projection of the CCSDT Hamiltonian on triples
        X[a, b, c, i, j, k] = < ijkabc | (H_N exp(T1+T2+T3))_C | 0 >
    """

    # h(vvov) and h(vooo) intermediates resulting from exp(-T1) H_N exp(T1)
    Q1 = -np.einsum("mnfe,an->amef", g[o, o, v, v], t1, optimize=True)
    I_vovv = g[v, o, v, v] + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    I_ooov = g[o, o, o, v] + 0.5 * Q1

    Q1 = -np.einsum("bmfe,am->abef", I_vovv, t1, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    I_vvvv = g[v, v, v, v] + Q1

    Q1 = +np.einsum("nmje,ei->mnij", I_ooov, t1, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    I_oooo = g[o, o, o, o] + Q1

    Q1 = g[v, o, o, v] + 0.5 * np.einsum("amef,ei->amif", g[v, o, v, v], t1, optimize=True)
    Q1 = np.einsum("amif,fj->amij", Q1, t1, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    I_vooo = g[v, o, o, o] + Q1 - np.einsum("nmij,an->amij", I_oooo, t1, optimize=True)
    # Added in for ROHF
    I_vooo += np.einsum("me,aeij->amij", f[o, v], t2, optimize=True)

    Q1 = g[o, v, o, v] - 0.5 * np.einsum("mnie,bn->mbie", g[o, o, o, v], t1, optimize=True)
    Q1 = -np.einsum("mbie,am->abie", Q1, t1, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    I_vvov = g[v, v, o, v] + Q1 + np.einsum("abfe,fi->abie", I_vvvv, t1, optimize=True)
    # Added in for ROHF
    #I_vvov -= np.einsum("me,abim->abie", f[o, v], t2, optimize=True)

    triples_res = -0.25 * np.einsum("amij,bcmk->abcijk", I_vooo, t2, optimize=True)
    triples_res += 0.25 * np.einsum("abie,ecjk->abcijk", I_vvov, t2, optimize=True)

    triples_res -= np.transpose(triples_res, (0, 1, 2, 3, 5, 4)) # (jk)
    triples_res -= np.transpose(triples_res, (0, 1, 2, 4, 3, 5)) + np.transpose(triples_res, (0, 1, 2, 5, 4, 3)) # (i/jk)
    triples_res -= np.transpose(triples_res, (0, 2, 1, 3, 4, 5)) # (bc)
    triples_res -= np.transpose(triples_res, (2, 1, 0, 3, 4, 5)) + np.transpose(triples_res, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return triples_res * e_abcijk

def kernel(fock, g, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core, use_quasi):
    """Solve the CCSDT system of nonlinear equations using Jacobi iterations
    with DIIS acceleration. The initial values of the T amplitudes are taken to be 0."""

    #eps = np.kron(np.diagonal(fock)[::2], np.ones(2))
    eps = np.diagonal(fock)
    n = np.newaxis
    e_abcijk = 1.0 / (- eps[v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                    + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o] + energy_shift )
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

    old_energy = cc_energy(t1, t2, fock, g, o, v)

    print("    ==> CC3 amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(maxit):

        tic = time.time()

        # Compute T3 using the perturbative approximation of CC3
        t3 = compute_t3(t1, t2, e_abcijk, fock, g, o, v)
        I_vooo, I_vvov = compute_ccs_intermediates(fock, g, t1, t2, o, v)
        residual_singles = singles_residual(t1, t2, t3, fock, g, o, v, I_vooo, I_vvov, e_abc)
        residual_doubles = doubles_residual(t1, t2, t3, fock, g, o, v, I_vooo, I_vvov, e_abc)

        res_norm = np.linalg.norm(residual_singles) + np.linalg.norm(residual_doubles)

        t1 += residual_singles * e_ai
        t2 += residual_doubles * e_abij

        current_energy = cc_energy(t1, t2, fock, g, o, v)
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
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(idx, current_energy, delta_e, res_norm, minutes, seconds))
    else:
        raise ValueError("CC3 iterations did not converge")

    diis_engine.cleanup()
    e_corr = cc_energy(t1, t2, fock, g, o, v)
    t3 = compute_t3(t1, t2, e_abcijk, fock, g, o, v)

    return (t1, t2, t3), e_corr


