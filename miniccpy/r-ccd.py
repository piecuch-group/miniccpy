import time
import numpy as np
from miniccpy.energy import rccd_energy
from miniccpy.diis import DIIS

def doubles_residual(t2, f, g, o, v):
    """Compute the projection of the CCD Hamiltonian on doubles
        X[a, b, i, j] = < ijab | (H_N exp(T2))_C | 0 >
    """
    t2s = t2 - np.transpose(t2, (0, 1, 3, 2))
    gs = g - np.transpose(g, (0, 1, 3, 2))
    # intermediates
    I_vv = (
        f[v, v]
        - 0.5 * np.einsum("mnef,afmn->ae", gs[o, o, v, v], t2s, optimize=True)
        - np.einsum("mnef,afmn->ae", g[o, o, v, v], t2, optimize=True)
    )
    I_oo = (
        f[o, o]
        + 0.5 * np.einsum("mnef,efin->mi", gs[o, o, v, v], t2s, optimize=True)
        + np.einsum("mnef,efin->mi", g[o, o, v, v], t2, optimize=True)
    )
    I_voov = (
        g[v, o, o, v]
        + np.einsum("mnef,aeim->anif", g[o, o, v, v], t2s, optimize=True)
        + np.einsum("mnef,aeim->anif", gs[o, o, v, v], t2, optimize=True)
    )
    Is_voov = (
        gs[v, o, o, v]
        + np.einsum("mnef,aeim->anif", gs[o, o, v, v], t2s, optimize=True)
        + np.einsum("nmfe,aeim->anif", g[o, o, v, v], t2, optimize=True)
    )
    I_oooo = g[o, o, o, o] + np.einsum("mnef,efij->mnij", g[o, o, v, v], t2, optimize=True)
    I_vovo = g[v, o, v, o] - np.einsum("mnef,afmj->anej", g[o, o, v, v], t2, optimize=True)

    doubles_res = np.einsum("ae,ebij->abij", I_vv, t2, optimize=True)
    doubles_res += np.einsum("be,aeij->abij", I_vv, t2, optimize=True)
    doubles_res -= np.einsum("mi,abmj->abij", I_oo, t2, optimize=True)
    doubles_res -= np.einsum("mj,abim->abij", I_oo, t2, optimize=True)
    doubles_res += np.einsum("amie,ebmj->abij", Is_voov, t2, optimize=True)
    doubles_res += np.einsum("amie,ebmj->abij", I_voov, t2s, optimize=True)
    doubles_res += np.einsum("mbej,aeim->abij", g[o, v, v, o], t2s, optimize=True)
    doubles_res += np.einsum("bmje,aeim->abij", gs[v, o, o, v], t2, optimize=True)
    doubles_res -= np.einsum("mbie,aemj->abij", g[o, v, o, v], t2, optimize=True)
    doubles_res -= np.einsum("amej,ebim->abij", I_vovo, t2, optimize=True)
    doubles_res += np.einsum("mnij,abmn->abij", I_oooo, t2, optimize=True)
    doubles_res += np.einsum("abef,efij->abij", g[v, v, v, v], t2, optimize=True)
    doubles_res += g[v, v, o, o]

    return doubles_res

def kernel(fock, g, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core, use_quasi):
    """Solve the R-CCD system of nonlinear equations using Jacobi iterations
    with DIIS acceleration. The initial values of the T amplitudes are taken to be 0."""

    eps = np.diagonal(fock)
    n = np.newaxis
    e_abij = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o] + energy_shift)

    nunocc, nocc = fock[v, o].shape
    n2 = nocc**2 * nunocc**2
    ndim = n2

    diis_engine = DIIS(ndim, diis_size, out_of_core)

    t2 = np.zeros((nunocc, nunocc, nocc, nocc))
    old_energy = rccd_energy(t2, g, o, v)

    print("    ==> R-CCD amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(maxit):

        tic = time.time()

        residual_doubles = doubles_residual(t2, fock, g, o, v)
        res_norm = np.linalg.norm(residual_doubles)
        t2 += residual_doubles * e_abij

        current_energy = rccd_energy(t2, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < convergence and res_norm < convergence:
            break

        if idx >= n_start_diis:
            diis_engine.push( (t2), (residual_doubles), idx) 
        if idx >= diis_size + n_start_diis:
            T_extrap = diis_engine.extrapolate()
            t2 = T_extrap.reshape((nunocc, nunocc, nocc, nocc))

        old_energy = current_energy

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(idx, current_energy, delta_e, res_norm, minutes, seconds))
    else:
        raise ValueError("CCD iterations did not converge")

    diis_engine.cleanup()
    e_corr = rccd_energy(t2, g, o, v)

    return (t2), e_corr


