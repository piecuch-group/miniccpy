import time
import numpy as np
from miniccpy.diis import DIIS
from miniccpy.utilities import get_memory_usage

def lrcc_energy(eta1, eta2, t1, t2, H1, H2, W, o, v):

    energy = np.einsum('ia,ai->', H1[o, v], eta1)
    energy += 0.25 * np.einsum('ijab,abij->', H2[o, o, v, v], eta2)
    energy += np.einsum("ia,ai->", W[o, v], t1)
    return energy

def singles_residual(eta1, eta2, t1, t2, H1, H2, W, o, v):
    """Compute the projection of the CCSD Hamiltonian on singles
        X[a, i] = < ia | [HBar, eta1 + eta2] + Wbar | 0 >
    """
    # < ia | [HBar, eta1 + eta2] | 0 >
    singles_res = -np.einsum("mi,am->ai", H1[o, o], eta1, optimize=True)
    singles_res += np.einsum("ae,ei->ai", H1[v, v], eta1, optimize=True)
    singles_res += np.einsum("amie,em->ai", H2[v, o, o, v], eta1, optimize=True)
    singles_res -= 0.5 * np.einsum("mnif,afmn->ai", H2[o, o, o, v], eta2, optimize=True)
    singles_res += 0.5 * np.einsum("anef,efin->ai", H2[v, o, v, v], eta2, optimize=True)
    singles_res += np.einsum("me,aeim->ai", H1[o, v], eta2, optimize=True)
    # < ia | Wbar | 0 >
    x_oo = W[o, o] + np.einsum("me,ei->mi", W[o, v], t1, optimize=True)
    singles_res += W[v, o].copy()
    singles_res -= np.einsum("mi,am->ai", x_oo, t1, optimize=True)
    singles_res += np.einsum("ae,ei->ai", W[v, v], t1, optimize=True)
    singles_res += np.einsum("me,aeim->ai", W[o, v], t2, optimize=True)
    return singles_res

def doubles_residual(eta1, eta2, t1, t2, H1, H2, W, o, v):
    """Compute the projection of the CCSD Hamiltonian on doubles
        X[a, b, i, j] = < ijab | [HBar, eta1 + eta2] + Wbar | 0 >
    """
    # < ijab | [HBar, eta1 + eta2] | 0 >
    doubles_res = -0.5 * np.einsum("mi,abmj->abij", H1[o, o], eta2, optimize=True)  # A(ij)
    doubles_res += 0.5 * np.einsum("ae,ebij->abij", H1[v, v], eta2, optimize=True)  # A(ab)
    doubles_res += 0.5 * 0.25 * np.einsum("mnij,abmn->abij", H2[o, o, o, o], eta2, optimize=True)
    doubles_res += 0.5 * 0.25 * np.einsum("abef,efij->abij", H2[v, v, v, v], eta2, optimize=True)
    doubles_res += np.einsum("amie,ebmj->abij", H2[v, o, o, v], eta2, optimize=True)  # A(ij)A(ab)
    doubles_res -= 0.5 * np.einsum("bmji,am->abij", H2[v, o, o, o], eta1, optimize=True)  # A(ab)
    doubles_res += 0.5 * np.einsum("baje,ei->abij", H2[v, v, o, v], eta1, optimize=True)  # A(ij)
    Q1 = -0.5 * np.einsum("mnef,bfmn->eb", H2[o, o, v, v], eta2, optimize=True)
    doubles_res += 0.5 * np.einsum("eb,aeij->abij", Q1, t2, optimize=True)  # A(ab)
    Q1 = 0.5 * np.einsum("mnef,efjn->mj", H2[o, o, v, v], eta2, optimize=True)
    doubles_res -= 0.5 * np.einsum("mj,abim->abij", Q1, t2, optimize=True)  # A(ij)
    Q1 = np.einsum("amfe,em->af", H2[v, o, v, v], eta1, optimize=True)
    doubles_res += 0.5 * np.einsum("af,fbij->abij", Q1, t2, optimize=True)  # A(ab)
    Q2 = np.einsum("nmie,em->ni", H2[o, o, o, v], eta1, optimize=True)
    doubles_res -= 0.5 * np.einsum("ni,abnj->abij", Q2, t2, optimize=True)  # A(ij)
    # < ijab | Wbar | 0 >
    x_oo = W[o, o] + np.einsum("me,ei->mi", W[o, v], t1, optimize=True)
    x_vv = W[v, v] - np.einsum("me,am->ae", W[o, v], t1, optimize=True)
    doubles_res -= 0.5 * np.einsum("mi,abmj->abij", x_oo, t2, optimize=True)
    doubles_res += 0.5 * np.einsum("ae,ebij->abij", x_vv, t2, optimize=True)
    doubles_res -= np.transpose(doubles_res, (0, 1, 3, 2))
    doubles_res -= np.transpose(doubles_res, (1, 0, 2, 3))
    return doubles_res

def kernel(T, H1, H2, W, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core):
    """Solve the CCSD system of nonlinear equations using Jacobi iterations
    with DIIS acceleration. The initial values of the T amplitudes are taken to be 0."""

    eps = np.diagonal(H1)
    n = np.newaxis
    e_abij = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o] + energy_shift)
    e_ai = 1.0 / (-eps[v, n] + eps[n, o] + energy_shift)

    t1, t2 = T

    nunocc, nocc = e_ai.shape
    n1 = nocc * nunocc
    n2 = nocc**2 * nunocc**2
    ndim = n1 + n2

    diis_engine = DIIS(ndim, diis_size, out_of_core)

    eta1 = np.zeros((nunocc, nocc))
    eta2 = np.zeros((nunocc, nunocc, nocc, nocc))
    old_energy = lrcc_energy(eta1, eta2, t1, t2, H1, H2, W, o, v)

    print("    ==> LR-CCSD(1) amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|     Wall Time     Memory")
    for idx in range(maxit):

        tic = time.time()

        residual_singles = singles_residual(eta1, eta2, t1, t2, H1, H2, W, o, v)
        residual_doubles = doubles_residual(eta1, eta2, t1, t2, H1, H2, W, o, v)

        res_norm = np.linalg.norm(residual_singles) + np.linalg.norm(residual_doubles)

        eta1 += residual_singles * e_ai
        eta2 += residual_doubles * e_abij

        current_energy = lrcc_energy(eta1, eta2, t1, t2, H1, H2, W, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < convergence and res_norm < convergence:
            break

        if idx >= n_start_diis:
            diis_engine.push( (eta1, eta2), (residual_singles, residual_doubles), idx) 
        if idx >= diis_size + n_start_diis:
            eta_extrap = diis_engine.extrapolate()
            eta1 = eta_extrap[:n1].reshape((nunocc, nocc))
            eta2 = eta_extrap[n1:].reshape((nunocc, nunocc, nocc, nocc))

        old_energy = current_energy

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(idx, current_energy, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        raise ValueError("LR-CCSD(1) iterations did not converge")

    diis_engine.cleanup()
    e_corr = lrcc_energy(eta1, eta2, t1, t2, H1, H2, W, o, v)

    return (eta1, eta2), e_corr


