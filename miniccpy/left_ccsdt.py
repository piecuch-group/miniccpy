import time
import numpy as np
from miniccpy.energy import lccsd_energy as lcc_energy
from miniccpy.diis import DIIS

def get_ccsdt_intermediates(l2, l3, t2, t3, o, v):
    nu, _, no, _ = t2.shape
    X1 = {"vo": [], "oo": [], "vv": []}
    X2 = {"ooov": [], "vovv": [],
          "vvvv": [], "oooo": [], "voov": []}

    # (L2 * T3)_C
    X1["vo"] = 0.25 * np.einsum("efmn,aefimn->ai", l2, t3, optimize=True)

    # (L3 * T2)_C
    X2["ooov"] = 0.5 * np.einsum('aefijn,efmn->jima', l3, t2, optimize=True)
    X2["vovv"] = -0.5 * np.einsum('abfimn,efmn->eiba', l3, t2, optimize=True)

    # (L3 * T3)_C
    X1["oo"] = (1.0 / 12.0) * np.einsum("efgmno,efgino->mi", l3, t3, optimize=True)
    X1["vv"] = -(1.0 / 12.0) * np.einsum("efgmno,afgmno->ae", l3, t3, optimize=True)
    X2["oooo"] = (1.0 / 6.0) * np.einsum("efgmno,efgijo->mnij", l3, t3, optimize=True)
    X2["vvvv"] = (1.0 / 6.0) * np.einsum("efgmno,abgmno->abef", l3, t3, optimize=True)
    X2["voov"] = 0.25 * np.einsum("efgmno,afgino->amie", l3, t3, optimize=True)
    return X1, X2

def LH_singles(l1, l2, l3, t2, t3, H1, H2, X1, X2, o, v):
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

    # 4-body Hbar
    I_vo = (
          -0.5 * np.einsum("nomg,egno->em", X2["ooov"], t2, optimize=True)
    )
    LH += np.einsum("em,imae->ai", I_vo, H2[o, o, v, v], optimize=True)

    LH -= np.einsum("nm,mina->ai", X1["oo"], H2[o, o, o, v], optimize=True)
    LH -= np.einsum("ef,fiea->ai", X1["vv"], H2[v, o, v, v], optimize=True)

    # < 0 | L3 * H(2) + L3 * (H(2) * T3)_C | ia >
    LH += np.einsum("ie,ea->ai", H1[o, v], X1["vv"], optimize=True)
    LH -= np.einsum("ma,im->ai", H1[o, v], X1["oo"], optimize=True)

    LH += 0.5 * np.einsum("nmoa,iomn->ai", X2["ooov"], H2[o, o, o, o], optimize=True)
    LH += np.einsum("fmae,eimf->ai", X2["vovv"], H2[v, o, o, v], optimize=True)

    LH -= 0.5 * np.einsum("gife,efag->ai", X2["vovv"], H2[v, v, v, v], optimize=True)
    LH -= np.einsum("imne,enma->ai", X2["ooov"], H2[v, o, o, v], optimize=True)

    LH += 0.5 * np.einsum("nmoa,iomn->ai", H2[o, o, o, v], X2["oooo"], optimize=True)
    LH += np.einsum("fmae,eimf->ai", H2[v, o, v, v], X2["voov"], optimize=True)

    LH -= 0.5 * np.einsum("gife,efag->ai", H2[v, o, v, v], X2["vvvv"], optimize=True)
    LH -= np.einsum("imne,enma->ai", H2[o, o, o, v], X2["voov"], optimize=True)

    LH += H1[o, v].transpose(1, 0)
    return LH

def LH_doubles(l1, l2, l3, t2, t3, H1, H2, X1, X2, o, v):
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

    # < 0 | L3 * H(2) | ijab >
    LH -= np.einsum("ejfb,fiea->abij", X2["vovv"], H2[v, o, v, v], optimize=True) # 1
    LH -= np.einsum("njmb,mina->abij", X2["ooov"], H2[o, o, o, v], optimize=True) # 2
    LH -= 0.25 * np.einsum("enab,jine->abij", X2["vovv"], H2[o, o, o, v], optimize=True) # 3
    LH -= 0.25 * np.einsum("jine,enab->abij", X2["ooov"], H2[v, o, v, v], optimize=True) # 4

    # < 0 | L3 * (H(2) * T3) | ijab >
    LH += np.einsum("ejmb,imae->abij", X2["voov"], H2[o, o, v, v], optimize=True) # 1
    LH += 0.125 * np.einsum("efab,ijef->abij", X2["vvvv"], H2[o, o, v, v], optimize=True) # 3
    LH += 0.125 * np.einsum("ijmn,mnab->abij", X2["oooo"], H2[o, o, v, v], optimize=True) # 4

    # 4-body HBar
    LH += 0.5 * np.einsum("ea,ijeb->abij", X1["vv"], H2[o, o, v, v], optimize=True) # 1
    LH -= 0.5 * np.einsum("im,jmba->abij", X1["oo"], H2[o, o, v, v], optimize=True) # 2

    # Moment-like terms
    LH += 0.25 * np.einsum("ebfijn,fena->abij", l3, H2[v, v, o, v], optimize=True) # 1
    LH -= 0.25 * np.einsum("abfmjn,finm->abij", l3, H2[v, o, o, o], optimize=True) # 3

    LH += 0.25 * H2[o, o, v, v].transpose(2, 3, 0, 1)

    LH -= np.transpose(LH, (1, 0, 2, 3))
    LH -= np.transpose(LH, (0, 1, 3, 2))
    return LH

def LH_triples(l1, l2, l3, t2, t3, H1, H2, X1, X2, o, v):
    """Compute the projection of the CCSDT Hamiltonian on doubles
        X[a, b, c, i, j, k] = < ijkabc | (H_N exp(T1+T2+T3))_C | 0 >
    """
    # < 0 | L1 * H(2) | ijkabc >
    LH = (9.0 / 36.0) * np.einsum("ai,jkbc->abcijk", l1, H2[o, o, v, v], optimize=True)

    # < 0 | L2 * H(2) | ijkabc >
    LH += (9.0 / 36.0) * np.einsum("bcjk,ia->abcijk", l2, H1[o, v], optimize=True)

    LH += (9.0 / 36.0) * np.einsum("ebij,ekac->abcijk", l2, H2[v, o, v, v], optimize=True)
    LH -= (9.0 / 36.0) * np.einsum("abmj,ikmc->abcijk", l2, H2[o, o, o, v], optimize=True)

    # < 0 | L3 * H(2) | ijkabc >
    LH += (3.0 / 36.0) * np.einsum("ea,ebcijk->abcijk", H1[v, v], l3, optimize=True)
    LH -= (3.0 / 36.0) * np.einsum("im,abcmjk->abcijk", H1[o, o], l3, optimize=True)
    LH += (9.0 / 36.0) * np.einsum("eima,ebcmjk->abcijk", H2[v, o, o, v], l3, optimize=True)
    LH += (3.0 / 72.0) * np.einsum("ijmn,abcmnk->abcijk", H2[o, o, o, o], l3, optimize=True)
    LH += (3.0 / 72.0) * np.einsum("efab,efcijk->abcijk", H2[v, v, v, v], l3, optimize=True)

    LH += (9.0 / 36.0) * np.einsum("ijeb,ekac->abcijk", H2[o, o, v, v], X2["vovv"], optimize=True)
    LH -= (9.0 / 36.0) * np.einsum("mjab,ikmc->abcijk", H2[o, o, v, v], X2["ooov"], optimize=True)

    LH -= np.transpose(LH, (0, 1, 2, 3, 5, 4)) # (jk)
    LH -= np.transpose(LH, (0, 1, 2, 4, 3, 5)) + np.transpose(LH, (0, 1, 2, 5, 4, 3)) # (i/jk)
    LH -= np.transpose(LH, (0, 2, 1, 3, 4, 5)) # (bc)
    LH -= np.transpose(LH, (2, 1, 0, 3, 4, 5)) + np.transpose(LH, (1, 0, 2, 3, 4, 5)) # (a/bc)
    return LH

def kernel(T, fock, H1, H2, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core):
    """Solve the left-CCSD system of nonlinear equations using Jacobi iterations
    with DIIS acceleration. The initial values of the L amplitudes are taken as T."""

    omega = 0.0
    eps = np.diagonal(H1)
    n = np.newaxis
    e_abcijk = 1.0 / (eps[v, n, n, n, n, n] + eps[n, v, n, n, n, n] + eps[n, n, v, n, n, n]
                    - eps[n, n, n, o, n, n] - eps[n, n, n, n, o, n] - eps[n, n, n, n, n, o] - omega + energy_shift )
    e_abij = 1.0 / (eps[v, n, n, n] + eps[n, v, n, n] - eps[n, n, o, n] - eps[n, n, n, o] - omega + energy_shift)
    e_ai = 1.0 / (eps[v, n] - eps[n, o] - omega + energy_shift)
    t1, t2, t3 = T

    nunocc, nocc = e_ai.shape
    n1 = nocc * nunocc
    n2 = nocc**2 * nunocc**2
    n3 = nocc**3 * nunocc**3
    ndim = n1 + n2 + n3

    diis_engine = DIIS(ndim, diis_size, out_of_core)

    l1 = t1.copy() 
    l2 = t2.copy()
    l3 = t3.copy()
    lh1 = np.zeros((nunocc, nocc))
    lh2 = np.zeros((nunocc, nunocc, nocc, nocc))
    lh3 = np.zeros((nunocc, nunocc, nunocc, nocc, nocc, nocc))

    old_energy = lcc_energy(l1, l2, lh1, lh2) + omega

    print("    ==> Left-CCSDT amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dL|")
    for idx in range(maxit):

        tic = time.time()

        # comptute L*T intermediates
        X1, X2 = get_ccsdt_intermediates(l2, l3, t2, t3, o, v)
        lh1 = LH_singles(l1, l2, l3, t2, t3, H1, H2, X1, X2, o, v)
        lh2 = LH_doubles(l1, l2, l3, t2, t3, H1, H2, X1, X2, o, v)
        lh3 = LH_triples(l1, l2, l3, t2, t3, H1, H2, X1, X2, o, v)

        lh1 = (omega * l1 - lh1) * e_ai
        lh2 = (omega * l2 - lh2) * e_abij
        lh3 = (omega * l3 - lh3) * e_abcijk
        l1 += lh1
        l2 += lh2
        l3 += lh3

        res_norm = np.linalg.norm(lh1.flatten()) + np.linalg.norm(lh2.flatten()) + np.linalg.norm(lh3.flatten())
        current_energy = lcc_energy(l1, l2, lh1, lh2) + omega
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < convergence and res_norm < convergence:
            break

        if idx >= n_start_diis:
            diis_engine.push( (l1, l2, l3), (lh1, lh2, lh3), idx) 
        if idx >= diis_size + n_start_diis:
            L_extrap = diis_engine.extrapolate()
            l1 = L_extrap[:n1].reshape((nunocc, nocc))
            l2 = L_extrap[n1:n1+n2].reshape((nunocc, nunocc, nocc, nocc))
            l3 = L_extrap[n1+n2:].reshape((nunocc, nunocc, nunocc, nocc, nocc, nocc))

        old_energy = current_energy

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(idx, current_energy, delta_e, res_norm, minutes, seconds))
    else:
        raise ValueError("left-CCSDT iterations did not converge")

    diis_engine.cleanup()
    e_corr = lcc_energy(l1, l2, lh1, lh2)

    return (l1, l2, l3), e_corr

