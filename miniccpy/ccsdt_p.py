import time
import numpy as np
from miniccpy.energy import cc_energy
from miniccpy.hbar import get_ccs_intermediates, get_ccsd_intermediates
from miniccpy.diis import DIIS
from miniccpy.lib import ccsdt_p_loops

def singles_residual(t1, t2, t3, t3_excitations, f, g, o, v, shift):
    """Compute the projection of the CCSDT Hamiltonian on singles
        X[a, i] = < ia | (H_N exp(T1+T2+T3))_C | 0 >
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
    singles_res += f[v, o]

    t1, singles_res = ccsdt_p_loops.ccsdt_p_loops.update_t1(
        t1, 
        singles_res,
        t3_excitations,
        t3, 
        g[o, o, v, v],
        f[o, o], f[v, v],
        shift
    )
    return t1, singles_res

def doubles_residual(t1, t2, t3, t3_excitations, f, g, o, v, shift):
    """Compute the projection of the CCSDT Hamiltonian on doubles
        X[a, b, i, j] = < ijab | (H_N exp(T1+T2+T3))_C | 0 >
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
    doubles_res += 0.25 * g[v, v, o, o]

    t2, doubles_res = ccsdt_p_loops.ccsdt_p_loops.update_t2(
        t2,
        doubles_res,
        t3_excitations,
        t3,
        H1[o, v],
        g[o, o, o, v] + H2[o, o, o, v], g[v, o, v, v] + H2[v, o, v, v],
        f[o, o], f[v, v],
        shift
    )
    return t2, doubles_res

def triples_residual(t1, t2, t3, t3_excitations, f, g, o, v, shift):
    """Compute the projection of the CCSDT Hamiltonian on triples
        X[a, b, c, i, j, k] = < ijkabc | (H_N exp(T1+T2+T3))_C | 0 >
    """

    H1, H2 = get_ccsd_intermediates(t1, t2, f, g, o, v)

    I2_vooo = H2[v, o, o, o] - np.einsum("me,aeij->amij", H1[o, v], t2, optimize=True)
    I2_vooo = I2_vooo.transpose(1, 0, 2, 3)

    triples_res, t3, t3_excitations = ccsdt_p_loops.ccsdt_p_loops.update_t3_p(
        t3, t3_excitations, 
        t2,
        H1[o, o], H1[v, v].T,
        g[o, o, v, v], H2[v, v, o, v].transpose(3, 0, 1, 2), I2_vooo,
        H2[o, o, o, o], H2[v, o, o, v].transpose(1, 3, 0, 2), H2[v, v, v, v].transpose(3, 2, 1, 0),
        f[o, o], f[v, v],
        shift
    )
    return t3, t3_excitations, triples_res


def kernel(fock, g, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core, use_quasi, t3_excitations):
    """Solve the CCSDT system of nonlinear equations using Jacobi iterations
    with DIIS acceleration. The initial values of the T amplitudes are taken to be 0."""

    # determine whether t3 updates should be done. Stupid compatibility with
    # empty sections of t3_excitations
    do_t3 = True
    if np.array_equal(t3_excitations[0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3 = False

    nunocc, nocc = fock[v, o].shape
    n1 = nocc * nunocc
    n2 = nocc**2 * nunocc**2
    n3 = t3_excitations.shape[0]
    ndim = n1 + n2 + n3

    diis_engine = DIIS(ndim, diis_size, out_of_core)

    t1 = np.zeros((nunocc, nocc))
    t2 = np.zeros((nunocc, nunocc, nocc, nocc))
    t3 = np.zeros(n3)

    old_energy = cc_energy(t1, t2, fock, g, o, v)

    print("    ==> CCSDT(P) amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(maxit):

        tic = time.time()

        t1, residual_singles = singles_residual(t1, t2, t3, t3_excitations, fock, g, o, v, energy_shift)
        t2, residual_doubles = doubles_residual(t1, t2, t3, t3_excitations, fock, g, o, v, energy_shift)
        if do_t3:
            t3, t3_excitations, residual_triples = triples_residual(t1, t2, t3, t3_excitations, fock, g, o, v, energy_shift)

        res_norm = np.linalg.norm(residual_singles) + np.linalg.norm(residual_doubles) + np.linalg.norm(residual_triples)

        current_energy = cc_energy(t1, t2, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < convergence and res_norm < convergence:
            break
 
        if idx >= n_start_diis:
            diis_engine.push((t1, t2, t3), (residual_singles, residual_doubles, residual_triples), idx)

        if idx >= diis_size + n_start_diis:
            T_extrap = diis_engine.extrapolate()
            t1 = T_extrap[:n1].reshape((nunocc, nocc))
            t2 = T_extrap[n1:n1+n2].reshape((nunocc, nunocc, nocc, nocc))
            t3 = T_extrap[n1+n2:]

        old_energy = current_energy

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(idx, current_energy, delta_e, res_norm, minutes, seconds))
    else:
        raise ValueError("CCSDT(P) iterations did not converge")

    diis_engine.cleanup()
    e_corr = cc_energy(t1, t2, fock, g, o, v)

    return (t1, t2, t3), e_corr


