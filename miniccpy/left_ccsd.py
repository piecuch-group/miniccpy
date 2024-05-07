import time
import numpy as np
from miniccpy.energy import lccsd_energy as lcc_energy
from miniccpy.diis import DIIS
from miniccpy.utilities import get_memory_usage

def LH_singles(l1, l2, t2, H1, H2, o, v):
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

    LH += H1[o, v].transpose(1, 0)

    return LH


def LH_doubles(l1, l2, t2, H1, H2, o, v):
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

    LH += 0.25 * H2[o, o, v, v].transpose(2, 3, 0, 1)

    LH -= np.transpose(LH, (1, 0, 2, 3))
    LH -= np.transpose(LH, (0, 1, 3, 2))

    return LH

def update(l1, l2, omega, e_ai, e_abij):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""

    l1 /= (omega - e_ai)
    l2 /= (omega - e_abij)

    return np.hstack([l1.flatten(), l2.flatten()])


def LH(l1, l2, t1, t2, H1, H2, o, v):
    """Compute the matrix-vector product H * R, where
    H is the CCSD similarity-transformed Hamiltonian and R is
    the EOMCCSD linear excitation operator."""

    # update L1
    LH1 = LH_singles(l1, l2, t2, H1, H2, o, v)
    # update L2
    LH2 = LH_doubles(l1, l2, t2, H1, H2, o, v)

    return np.hstack( [LH1.flatten(), LH2.flatten()] )


def kernel(T, fock, H1, H2, o, v, maxit, convergence, energy_shift, diis_size, n_start_diis, out_of_core):
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

    print("    ==> Left-CCSD amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dL|     Wall Time     Memory")
    for idx in range(maxit):

        tic = time.time()

        lh1 = LH_singles(l1, l2, t2, H1, H2, o, v)
        lh2 = LH_doubles(l1, l2, t2, H1, H2, o, v)

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
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(idx, current_energy, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        raise ValueError("left-CCSD iterations did not converge")

    diis_engine.cleanup()
    e_corr = lcc_energy(l1, l2, lh1, lh2)

    return (l1, l2), e_corr


def kernel_davidson(L0, T, omega, H1, H2, o, v, maxit=80, convergence=1.0e-07, max_size=20, nrest=1):
    """
    Diagonalize the similarity-transformed CCSD Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
    eps = np.diagonal(H1)
    n = np.newaxis
    e_abij = (eps[v, n, n, n] + eps[n, v, n, n] - eps[n, n, o, n] - eps[n, n, n, o])
    e_ai = (eps[v, n] - eps[n, o])

    t1, t2 = T

    nunocc, nocc = e_ai.shape
    n1 = nunocc * nocc
    n2 = nocc**2 * nunocc**2
    ndim = n1 + n2

    # Pad the initial guess vector to fill the dimension of the problem
    if len(L0) < ndim:
        L = np.zeros(ndim)
        L[:len(L0)] = L0
    else:
        L = L0.copy()

    # Allocate the B and sigma matrices
    sigma = np.zeros((ndim, max_size))
    B = np.zeros((ndim, max_size))
    restart_block = np.zeros((ndim, nrest))

    # Initial values
    B[:, 0] = L
    sigma[:, 0] = LH(L[:n1].reshape(nunocc, nocc),
                     L[n1:].reshape(nunocc, nunocc, nocc, nocc),
                     t1, t2, H1, H2, o, v)

    print("    ==> Left-EOMCCSD iterations <==")
    print("    The initial guess energy = ", omega)
    print("")
    print("     Iter               Energy                 |dE|                 |dR|")
    curr_size = 1
    for niter in range(maxit):
        tic = time.time()
        # store old energy
        omega_old = omega

        # solve projection subspace eigenproblem
        G = np.dot(B[:, :curr_size].T, sigma[:, :curr_size])
        e, alpha = np.linalg.eig(G)

        # select root based on maximum overlap with initial guess
        idx = np.argsort(abs(alpha[0, :]))
        alpha = np.real(alpha[:, idx[-1]])

        # Get the eigenpair of interest
        omega = np.real(e[idx[-1]])
        L = np.dot(B[:, :curr_size], alpha)
        restart_block[:, niter % nrest] = L

        # calculate residual vector
        residual = np.dot(sigma[:, :curr_size], alpha) - omega * L
        res_norm = np.linalg.norm(residual)
        delta_e = omega - omega_old

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(niter, omega, delta_e, res_norm, minutes, seconds))
        if res_norm < convergence and abs(delta_e) < convergence:
            break

        # update residual vector
        q = update(residual[:n1].reshape(nunocc, nocc),
                   residual[n1:].reshape(nunocc, nunocc, nocc, nocc),
                   omega,
                   e_ai,
                   e_abij)
        for p in range(curr_size):
            b = B[:, p] / np.linalg.norm(B[:, p])
            q -= np.dot(b.T, q) * b
        q *= 1.0 / np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[:, curr_size] = q
            sigma[:, curr_size] = LH(q[:n1].reshape(nunocc, nocc),
                                     q[n1:].reshape(nunocc, nunocc, nocc, nocc),
                                     t1, t2, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[:, j] = restart_block[:, j]
                sigma[:, j] = LH(restart_block[:n1, j].reshape(nunocc, nocc),
                                 restart_block[n1:, j].reshape(nunocc, nunocc, nocc, nocc),
                                 t1, t2, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1
    else:
        print("Left-EOMCCSD iterations did not converge")

    # Save the final converged root in an excitation tuple
    L = (L[:n1].reshape(nunocc, nocc), L[n1:].reshape(nunocc, nunocc, nocc, nocc))

    return L, omega
