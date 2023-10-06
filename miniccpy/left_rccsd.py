import time
import numpy as np
from miniccpy.energy import lccsd_energy as lcc_energy
from miniccpy.diis import DIIS

def LT_intermediates(l2, t2):
    """Compute L2*T2-type one-body intermediates."""
    # Allocate a dictionary to store the two intermediates
    I = {"vv": None, "oo": None}
    I["vv"] = (
          -2.0 * np.einsum("afmn,efmn->ea", l2, t2, optimize=True)
          + np.einsum("afnm,efmn->ea", l2, t2, optimize=True)
    )
    I["oo"] = (
          2.0 * np.einsum("efin,efjn->ij", l2, t2, optimize=True)
          - np.einsum("efni,efjn->ij", l2, t2, optimize=True)
    )
    return I

def LH_singles(l1, l2, t2, H1, H2, I, o, v):
    """Compute the projection of the CCSD Hamiltonian on singles
        X[a, i] = < 0 | (1 + L1 + L2)*(H_N exp(T1+T2))_C | ia >
    """
    LH = np.einsum("ea,ei->ai", H1[v, v], l1, optimize=True)
    LH -= np.einsum("im,am->ai", H1[o, o], l1, optimize=True)
    LH += 2.0 * np.einsum("eima,em->ai", H2[v, o, o, v], l1, optimize=True)
    LH -= np.einsum("eiam,em->ai", H2[v, o, v, o], l1, optimize=True)
    LH += 2.0 * np.einsum("fena,efin->ai", H2[v, v, o, v], l2, optimize=True)
    LH -= np.einsum("fena,efni->ai", H2[v, v, o, v], l2, optimize=True)
    LH -= 2.0 * np.einsum("finm,afmn->ai", H2[v, o, o, o], l2, optimize=True)
    LH += np.einsum("finm,afnm->ai", H2[v, o, o, o], l2, optimize=True)
    LH -= 2.0 * np.einsum("ge,eiga->ai", I["vv"], H2[v, o, v, v], optimize=True)
    LH += np.einsum("ge,eiag->ai", I["vv"], H2[v, o, v, v], optimize=True)
    LH -= 2.0 * np.einsum("mn,nima->ai", I["oo"], H2[o, o, o, v], optimize=True)
    LH += np.einsum("mn,inma->ai", I["oo"], H2[o, o, o, v], optimize=True)
    LH += H1[o, v].transpose(1, 0)
    return LH

def LH_doubles(l1, l2, t2, H1, H2, I, o, v):
    """Compute the projection of the CCSD Hamiltonian on doubles
        X[a, b, i, j] = < 0 | (1 + L2 + L2) * (H_N exp(T1+T2))_C | ijab >
    """
    LH = -np.einsum("ijmb,am->abij", H2[o, o, o, v], l1, optimize=True)
    LH += np.einsum("ejab,ei->abij", H2[v, o, v, v], l1, optimize=True)
    LH += 2.0 * np.einsum("ejmb,aeim->abij", H2[v, o, o, v], l2, optimize=True)
    LH -= np.einsum("ejmb,aemi->abij", H2[v, o, o, v], l2, optimize=True)
    LH += np.einsum("ea,ebij->abij", H1[v, v], l2, optimize=True)
    LH -= np.einsum("im,abmj->abij", H1[o, o], l2, optimize=True)
    LH += np.einsum("jb,ai->abij", H1[o, v], l1, optimize=True)
    LH += 0.5 * np.einsum("ijmn,abmn->abij", H2[o, o, o, o], l2, optimize=True)
    LH += 0.5 * np.einsum("efab,efij->abij", H2[v, v, v, v], l2, optimize=True)
    LH -= np.einsum("eiam,ebmj->abij", H2[v, o, v, o], l2, optimize=True)
    LH -= np.einsum("ejam,ebim->abij", H2[v, o, v, o], l2, optimize=True)
    LH += np.einsum("ea,ijeb->abij", I["vv"], H2[o, o, v, v], optimize=True)
    LH -= np.einsum("im,mjab->abij", I["oo"], H2[o, o, v, v], optimize=True)
    LH += 0.5 * H2[o, o, v, v].transpose(2, 3, 0, 1)
    # apply symmetrizer (ij)(ab)
    LH += LH.transpose(1, 0, 3, 2)
    return LH

def kernel(T, H1, H2, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core):
    """Solve the left-CCSD system of nonlinear equations using Jacobi iterations
    with DIIS acceleration. The initial values of the L amplitudes are taken as T."""

    omega = 0.0
    eps = np.diagonal(H1)
    n = np.newaxis
    e_abij = 1.0 / (eps[v, n, n, n] + eps[n, v, n, n] - eps[n, n, o, n] - eps[n, n, n, o] - omega + energy_shift)
    e_ai = 1.0 / (eps[v, n] - eps[n, o] - omega + energy_shift)
    t1, t2 = T

    nunocc, nocc = e_ai.shape
    n1 = nocc * nunocc
    n2 = nocc**2 * nunocc**2
    ndim = n1 + n2

    diis_engine = DIIS(ndim, diis_size, out_of_core)

    l1 = t1.copy() 
    l2 = t2.copy()
    lh1 = np.zeros((nunocc, nocc))
    lh2 = np.zeros((nunocc, nunocc, nocc, nocc))

    old_energy = lcc_energy(l1, l2, lh1, lh2) + omega

    print("    ==> Left-RCCSD amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dL|")
    for idx in range(maxit):

        tic = time.time()

        I = LT_intermediates(l2, t2)
        lh1 = LH_singles(l1, l2, t2, H1, H2, I, o, v)
        lh2 = LH_doubles(l1, l2, t2, H1, H2, I, o, v)

        lh1 = (omega * l1 - lh1) * e_ai
        lh2 = (omega * l2 - lh2) * e_abij
        l1 += lh1
        l2 += lh2

        res_norm = np.linalg.norm(lh1.flatten()) + np.linalg.norm(lh2.flatten())
        current_energy = lcc_energy(l1, l2, lh1, lh2) + omega
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < convergence and res_norm < convergence:
            break

        if idx >= n_start_diis:
            diis_engine.push( (l1, l2), (lh1, lh2), idx) 
        if idx >= diis_size + n_start_diis:
            L_extrap = diis_engine.extrapolate()
            l1 = L_extrap[:n1].reshape((nunocc, nocc))
            l2 = L_extrap[n1:].reshape((nunocc, nunocc, nocc, nocc))

        old_energy = current_energy

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(idx, current_energy, delta_e, res_norm, minutes, seconds))
    else:
        raise ValueError("left-CCSD iterations did not converge")

    diis_engine.cleanup()
    e_corr = lcc_energy(l1, l2, lh1, lh2)

    return (l1, l2), e_corr


