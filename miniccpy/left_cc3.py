import time
import numpy as np
from miniccpy.energy import lccsd_energy as lcc_energy
from miniccpy.diis import DIIS
from miniccpy.helper_cc3 import compute_l3, compute_leftcc3_intermediates

def get_ccsdt_intermediates(l2, l3, t2, t3, o, v):
    nu, _, no, _ = t2.shape
    X1 = {"vo": []}
    X2 = {"ooov": [], "vovv": []}
    # (L2 * T3)_C
    X1["vo"] = 0.25 * np.einsum("efmn,aefimn->ai", l2, t3, optimize=True)
    # (L3 * T2)_C
    X2["ooov"] = 0.5 * np.einsum('aefijn,efmn->jima', l3, t2, optimize=True)
    X2["vovv"] = -0.5 * np.einsum('abfimn,efmn->eiba', l3, t2, optimize=True)
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

def LH_doubles(l1, l2, l3, t1, t2, H1, H2, X1, X2, h_vvov, h_vooo, o, v):
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
    LH += 0.25 * np.einsum("ebfijn,fena->abij", l3, h_vvov, optimize=True) # 1
    LH -= 0.25 * np.einsum("abfmjn,finm->abij", l3, h_vooo, optimize=True) # 3

    LH += 0.25 * H2[o, o, v, v].transpose(2, 3, 0, 1)
    LH -= np.transpose(LH, (1, 0, 2, 3))
    LH -= np.transpose(LH, (0, 1, 3, 2))
    return LH

def LH_triples(l1, l2, l3, t1, t2, f, g, H1, H2, o, v):
    # < 0 | L1 * H(2) | ijkabc >
    LH = (9.0 / 36.0) * np.einsum("ai,jkbc->abcijk", l1, g[o, o, v, v], optimize=True)
    # < 0 | L2 * H(2) | ijkabc >
    LH += (9.0 / 36.0) * np.einsum("bcjk,ia->abcijk", l2, H1[o, v], optimize=True)
    LH += (9.0 / 36.0) * np.einsum("ebij,ekac->abcijk", l2, H2[v, o, v, v], optimize=True)
    LH -= (9.0 / 36.0) * np.einsum("abmj,ikmc->abcijk", l2, H2[o, o, o, v], optimize=True)
    #
    LH += (3.0 / 36.0) * np.einsum("ea,ebcijk->abcijk", f[v, v], l3, optimize=True)
    LH -= (3.0 / 36.0) * np.einsum("im,abcmjk->abcijk", f[o, o], l3, optimize=True)
    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    LH -= np.transpose(LH, (0, 1, 2, 3, 5, 4))
    LH -= np.transpose(LH, (0, 1, 2, 4, 3, 5)) + np.transpose(LH, (0, 1, 2, 5, 4, 3))
    LH -= np.transpose(LH, (0, 2, 1, 3, 4, 5))
    LH -= np.transpose(LH, (1, 0, 2, 3, 4, 5)) + np.transpose(LH, (2, 1, 0, 3, 4, 5))
    return LH


def LH(l1, l2, l3, t1, t2, t3, f, g, H1, H2, o, v):
    """Compute the matrix-vector product L * H, where
    H is the CCSDT similarity-transformed Hamiltonian and L is
    the EOMCCSDT linear de-excitation operator."""
    # comptute L*T intermediates
    X1, X2 = get_ccsdt_intermediates(l2, l3, t2, t3, o, v)
    # Get CCS intermediates (it would be nice to not have to recompute these in left-CC)
    h_vvov, h_vooo, h_voov, h_vvvv, h_oooo = compute_leftcc3_intermediates(t1, t2, f, g, o, v)
    # update L1
    LH1 = LH_singles(l1, l2, t1, t2, H1, H2, X1, X2, h_voov, h_vvvv, h_oooo, o, v)
    # update L2
    LH2 = LH_doubles(l1, l2, l3, t1, t2, H1, H2, X1, X2, h_vvov, h_vooo, o, v)
    # update L3
    LH3 = LH_triples(l1, l2, l3, t1, t2, f, g, H1, H2, o, v)
    return np.hstack( [LH1.flatten(), LH2.flatten(), LH3.flatten()] )

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
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(maxit):

        tic = time.time()

        # comptute L*T intermediates
        X1, X2 = get_ccsdt_intermediates(l2, l3, t2, t3, o, v)
        # Get CCS intermediates (it would be nice to not have to recompute these in left-CC)
        h_vvov, h_vooo, h_voov, h_vvvv, h_oooo = compute_leftcc3_intermediates(t1, t2, f, g, o, v)
        residual_singles = singles_residual(t1, t2, fock, g, o, v, I_vooo, I_vvov, e_abc)
        residual_doubles = doubles_residual(t1, t2, fock, g, o, v, I_vooo, I_vvov, e_abc)
        # update L1
        LH1 = LH_singles(l1, l2, t1, t2, H1, H2, X1, X2, h_voov, h_vvvv, h_oooo, o, v)
        # update L2
        LH2 = LH_doubles(l1, l2, l3, t1, t2, H1, H2, X1, X2, h_vvov, h_vooo, o, v)
        # update L3
        LH3 = LH_triples(l1, l2, l3, t1, t2, f, g, H1, H2, o, v)

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
        raise ValueError("left-CC3 iterations did not converge")

    diis_engine.cleanup()
    e_corr = cc_energy(t1, t2, fock, g, o, v)

    return (t1, t2), e_corr
