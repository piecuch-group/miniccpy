import time
import numpy as np
from miniccpy.energy import cc_energy, hf_energy, hf_energy_from_fock
from miniccpy.hbar import get_ccs_intermediates
from miniccpy.diis import DIIS

from miniccpy.updates import update_t2

def singles_residual(t1, t2, f, g, o, v):
    """Compute the projection of the CCSD Hamiltonian on singles
        X[a, i] = < ia | (H_N exp(T1+T2))_C | 0 >
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

    return singles_res


def doubles_residual(t1, t2, f, g, o, v):
    """Compute the projection of the CCSD Hamiltonian on doubles
        X[a, b, i, j] = < ijab | (H_N exp(T1+T2))_C | 0 >
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

    doubles_res -= np.transpose(doubles_res, (1, 0, 2, 3))
    doubles_res -= np.transpose(doubles_res, (0, 1, 3, 2))

    doubles_res += g[v, v, o, o]

    return doubles_res


def kernel(fock, g, o, v, maxit, convergence, shift, diis_size, n_start_diis, out_of_core, energy_shift):
    """Solve the CCSD system of nonlinear equations using Jacobi iterations
    with DIIS acceleration. The initial values of the T amplitudes are taken to be 0."""

    eps = np.diagonal(fock)
    n = np.newaxis
    e_abij = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o] + energy_shift)
    e_ai = 1.0 / (-eps[v, n] + eps[n, o] + energy_shift)

    nunocc, nocc = e_ai.shape
    n1 = nocc * nunocc
    n2 = nocc**2 * nunocc**2
    ndim = n1 + n2

    diis_engine = DIIS(ndim, diis_size, out_of_core)

    t1 = np.zeros((nunocc, nocc))
    t2 = np.zeros((nunocc, nunocc, nocc, nocc))
    old_energy = cc_energy(t1, t2, fock, g, o, v)

    print("    ==> CCSD amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(maxit):

        tic = time.time()

        residual_singles = singles_residual(t1, t2, fock, g, o, v)
        residual_doubles = doubles_residual(t1, t2, fock, g, o, v)

        res_norm = np.linalg.norm(residual_singles) + np.linalg.norm(residual_doubles)

        t1 += residual_singles * e_ai
        #t2 += residual_doubles * e_abij
        t2 = update_t2(t2, residual_doubles, fock, g, o, v, shift, quasi=True)

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
        raise ValueError("CCSD iterations did not converge")

    diis_engine.cleanup()
    e_corr = cc_energy(t1, t2, fock, g, o, v)

    return (t1, t2), e_corr


