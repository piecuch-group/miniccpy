import time
import numpy as np
from miniccpy.energy import cc_energy
from miniccpy.helper_cc import get_ccs_intermediates
from miniccpy.helper_cc3 import compute_cc3_intermediates
from miniccpy.diis import DIIS
from miniccpy.utilities import get_memory_usage

def singles_residual(t1, t2, f, g, o, v):
    """Compute the projection of the CCSDT Hamiltonian on singles
        X[a, i] = < ia | (H_N exp(T1+T2+T3))_C | 0 >
    """
    no, nu = f[o, v].shape
    # Intermediates
    chi_vv = f[v, v] + np.einsum("anef,fn->ae", g[v, o, v, v], t1, optimize=True)
    chi_oo = f[o, o] + np.einsum("mnif,fn->mi", g[o, o, o, v], t1, optimize=True)
    h_ov = f[o, v] + np.einsum("mnef,fn->me", g[o, o, v, v], t1, optimize=True)
    h_oo = chi_oo + np.einsum("me,ei->mi", h_ov, t1, optimize=True)
    h_ooov = g[o, o, o, v] + np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    h_vovv = g[v, o, v, v] - np.einsum("mnfe,an->amef", g[o, o, v, v], t1, optimize=True)
    # CCSD residual
    singles_res = -np.einsum("mi,am->ai", h_oo, t1, optimize=True)
    singles_res += np.einsum("ae,ei->ai", chi_vv, t1, optimize=True)
    singles_res += np.einsum("anif,fn->ai", g[v, o, o, v], t1, optimize=True)
    singles_res += np.einsum("me,aeim->ai", h_ov, t2, optimize=True)
    singles_res -= 0.5 * np.einsum("mnif,afmn->ai", h_ooov, t2, optimize=True)
    singles_res += 0.5 * np.einsum("anef,efin->ai", h_vovv, t2, optimize=True)
    singles_res += f[v, o]
    return singles_res

def doubles_residual(t1, t2, f, g, o, v):
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
    # CCSD residual parts
    doubles_res = -0.5 * np.einsum("amij,bm->abij", I_vooo, t1, optimize=True)
    doubles_res += 0.5 * np.einsum("abie,ej->abij", H2[v, v, o, v], t1, optimize=True)
    doubles_res += 0.5 * np.einsum("ae,ebij->abij", I_vv, t2, optimize=True)
    doubles_res -= 0.5 * np.einsum("mi,abmj->abij", I_oo, t2, optimize=True)
    doubles_res += np.einsum("amie,ebmj->abij", I_voov, t2, optimize=True)
    doubles_res += 0.25 * np.einsum("abef,efij->abij", g[v, v, v, v], tau, optimize=True)
    doubles_res += 0.125 * np.einsum("mnij,abmn->abij", I_oooo, t2, optimize=True)
    doubles_res += 0.25 * g[v, v, o, o]
    return doubles_res

def add_t3_contributions(singles_res, doubles_res, t1, t2, f, g, I_vooo, I_vvov, e_abc, o, v):
    # Compute additional CCS-like intermediates
    h_ooov = g[o, o, o, v] + np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True) 
    h_vovv = g[v, o, v, v] - np.einsum("mnfe,an->amef", g[o, o, v, v], t1, optimize=True) # no(2)nu(3)
    h_ov = f[o, v] + np.einsum("mnef,fn->me", g[o, o, v, v], t1, optimize=True)
    # get orbital dimensions
    nu, no = t1.shape
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                # fock denominator for occupied
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                # -1/2 A(k/ij)A(abc) I(amij) * t(bcmk)
                t3_abc = -0.5 * np.einsum("am,bcm->abc", I_vooo[:, :, i, j], t2[:, :, :, k], optimize=True)
                t3_abc += 0.5 * np.einsum("am,bcm->abc", I_vooo[:, :, k, j], t2[:, :, :, i], optimize=True)
                t3_abc += 0.5 * np.einsum("am,bcm->abc", I_vooo[:, :, i, k], t2[:, :, :, j], optimize=True)
                # 1/2 A(i/jk)A(abc) I(abie) * t(ecjk)
                t3_abc += 0.5 * np.einsum("abe,ec->abc", I_vvov[:, :, i, :], t2[:, :, j, k], optimize=True)
                t3_abc -= 0.5 * np.einsum("abe,ec->abc", I_vvov[:, :, j, :], t2[:, :, i, k], optimize=True)
                t3_abc -= 0.5 * np.einsum("abe,ec->abc", I_vvov[:, :, k, :], t2[:, :, j, i], optimize=True)
                # Antisymmetrize A(abc)
                t3_abc -= np.transpose(t3_abc, (1, 0, 2)) + np.transpose(t3_abc, (2, 1, 0)) # A(a/bc)
                t3_abc -= np.transpose(t3_abc, (0, 2, 1)) # A(bc)
                # Divide t_abc by the denominator
                t3_abc /= (denom_occ + e_abc)
                # Compute diagram: 1/2 A(i/jk) v(jkbc) * t(abcijk)
                singles_res[:, i] += 0.5 * np.einsum("bc,abc->a", g[o, o, v, v][j, k, :, :], t3_abc, optimize=True)
                singles_res[:, j] -= 0.5 * np.einsum("bc,abc->a", g[o, o, v, v][i, k, :, :], t3_abc, optimize=True)
                singles_res[:, k] -= 0.5 * np.einsum("bc,abc->a", g[o, o, v, v][j, i, :, :], t3_abc, optimize=True)
                # Compute diagram: A(ij) [A(k/ij) h(ke) * t3(abeijk)]
                doubles_res[:, :, i, j] += 0.5 * np.einsum("e,abe->ab", h_ov[k, :], t3_abc, optimize=True) # (1)
                doubles_res[:, :, j, k] += 0.5 * np.einsum("e,abe->ab", h_ov[i, :], t3_abc, optimize=True) # (ik)
                doubles_res[:, :, i, k] -= 0.5 * np.einsum("e,abe->ab", h_ov[j, :], t3_abc, optimize=True) # (jk)
                # Compute diagram: -A(j/ik) h(ik:f) * t3(abfijk)
                doubles_res[:, :, :, j] -= 0.5 * np.einsum("mf,abf->abm", h_ooov[i, k, :, :], t3_abc, optimize=True)
                doubles_res[:, :, :, i] += 0.5 * np.einsum("mf,abf->abm", h_ooov[j, k, :, :], t3_abc, optimize=True)
                doubles_res[:, :, :, k] += 0.5 * np.einsum("mf,abf->abm", h_ooov[i, j, :, :], t3_abc, optimize=True)
                # Compute diagram: 1/2 A(k/ij) h(akef) * t3(ebfijk)
                doubles_res[:, :, i, j] += 0.5 * np.einsum("aef,ebf->ab", h_vovv[:, k, :, :], t3_abc, optimize=True)
                doubles_res[:, :, j, k] += 0.5 * np.einsum("aef,ebf->ab", h_vovv[:, i, :, :], t3_abc, optimize=True)
                doubles_res[:, :, i, k] -= 0.5 * np.einsum("aef,ebf->ab", h_vovv[:, j, :, :], t3_abc, optimize=True)
    # Antisymmetrize
    doubles_res -= np.transpose(doubles_res, (1, 0, 2, 3))
    doubles_res -= np.transpose(doubles_res, (0, 1, 3, 2))
    # Manually clear all diagonal elements
    for a in range(nu):
        doubles_res[a, a, :, :] *= 0.0
    for i in range(no):
        doubles_res[:, :, i, i] *= 0.0
    return singles_res, doubles_res

def kernel(fock, g, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core, use_quasi):
    """Solve the CCSDT system of nonlinear equations using Jacobi iterations
    with DIIS acceleration. The initial values of the T amplitudes are taken to be 0."""

    #eps = np.kron(np.diagonal(fock)[::2], np.ones(2))
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

    old_energy = cc_energy(t1, t2, fock, g, o, v)

    print("    ==> CC3 amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|     Wall Time     Memory")
    for idx in range(maxit):

        tic = time.time()

        # Compute T3 using the perturbative approximation of CC3
        I_vooo, I_vvov = compute_cc3_intermediates(fock, g, t1, t2, o, v)
        residual_singles = singles_residual(t1, t2, fock, g, o, v)
        residual_doubles = doubles_residual(t1, t2, fock, g, o, v)
        residual_singles, residual_doubles = add_t3_contributions(residual_singles, residual_doubles,
                                                                  t1, t2, fock, g, I_vooo, I_vvov, e_abc, o, v)

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
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(idx, current_energy, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        raise ValueError("CC3 iterations did not converge")

    diis_engine.cleanup()
    e_corr = cc_energy(t1, t2, fock, g, o, v)

    return (t1, t2), e_corr

#####################################################################
# AN IMPORTANT NOTE ABOUT ROHF-BASED CC3 (AND EOMCC3) CALCULATIONS: #
#####################################################################
# As explained in JCP 122 054110 (2005), the Fock matrix corresponding to
# ROHF orbitals is NOT diagonal in the occupied-occupied, virtual-virtual,
# and occupied-virtual blocks. This results in 3 complications:
#
# (1) Direct evaluation of T3 using MP denominator no longer holds (see explanation below).
# (2) f(o,v) terms enter the equations. This can result in the addition of
#     formerly neglected terms that enter as 2nd-order in wave function,
#     assuming that F and T1 are 0th order. These terms include
#     < ijkabc | (F_N * T2**2)_C | 0>, which is a moment-like term and easy to
#     deal with, but also brings about the third complication, namely, terms like
# (3) < ijkabc | (F_N*T1*T3)_C | 0 >. This term is 2nd-order in wave function and non-zero since
#     f(o,v) is non-zero. This term cannot be dealt with easily, so ROHF-CC3 is DEFINED
#     to neglect this term.
#
# Explanation of (1):
# The fact that f(o,o) and f(v,v) are no longer diagonal comes into play
# when we consider the direct evaluation of T3 from its
# projection after invoking the simplifications characteristic of CC3,
# < ijkabc | (F_N * T3)_C | 0 > + < ijkabc | (H(1)*T2)_C | 0 > = 0.
# Evaluating the contraction with the Fock matrix produces
# -A(i/jk) f(mi)*t(abcijk) + A(a/bc) f(ae)*t(abcijk) = -<ijkabc | (H(1)*T2)_C | 0 >,
# and after separating out diagonal and non-diagonal components of the Fock matrix,
# we get D_MP(abcijk)*t(abcijk) + D_ND(abcijk)(t) = -<ijkabc | (H(1)*T2)_C | 0 >,
# where D_MP(abcijk) = e_a + e_b + e_c - e_i - e_j - e_k is the usual MP denominator and
# D_ND(abcijk)(t) = A(i/jk) [1 - delta(mi)]*f(mi)*t(abcijk) + A(a/bc)[1 - delta(ae)]*f(ae)*t(abcijk)
# denotes the part of <ijkabc|(F_N*T3)_C|0> resulting from non-diagonal portions of the
# Fock matrix in the occupied-occupied and virtual-virtual blocks.
# For RHF orbitals, f(mi) = e_i delta(mi) and f(ae) = e_a delta(ae), so the
# non-diagonal contributions from the Fock matrix are 0, and we recover the usual
# CC3-type expression for T3, namely,
# t3(abcijk) = -<ijkabc|(H(1)*T2)_C|0> / D_MP(abcijk).
# For ROHF orbitals, however, the D_ND terms are NON-ZERO. This means that the above
# approximation for T3 is not strictly correct, and we can see this discrepancy between the results
# produced with "cc3-full" and "cc3". As far as I can tell, there is no standard way to resolve
# this issue. You cannot incorporate the non-diagonal terms because that would require storing
# T3, thus destroying the point of the method. In JCP 122 054110 (2005) (which should correspond
# to the implementation in Psi4), they semi-canonicalize the ROHF orbitals before entering CC3, so
# that the Fock matrix is made diagonal in the occ-occ and virt-virt blocks. Thus, they can save
# the structure of the ROHF-CC3 method (note that, they also mention that this type of trick is typically
# used in ROHF-CCSD(T) calculations as well) at the expense of using a UHF-CC3-type implementation.
