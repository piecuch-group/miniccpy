import time
import numpy as np
from miniccpy.utilities import get_memory_usage

def kernel(R0, T, omega, H1, H2, o, v, maxit=80, convergence=1.0e-07, max_size=20, nrest=1):
    """
    Diagonalize the similarity-transformed CCSD Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
    from miniccpy.energy import calc_rel_dea

    eps = np.diagonal(H1)
    n = np.newaxis
    e_ab = (eps[v, n] + eps[n, v])

    t1, t2 = T

    nunocc, nocc = t1.shape
    n1 = nunocc**2
    ndim = n1
    
    if len(R0) <= ndim:
        R = np.zeros(ndim)
        R[:len(R0)] = R0

    # Allocate the B and sigma matrices
    sigma = np.zeros((ndim, max_size))
    B = np.zeros((ndim, max_size))
    restart_block = np.zeros((ndim, nrest))

    # Initial values
    B[:, 0] = R
    sigma[:, 0] = HR(R.reshape(nunocc, nunocc),
                     t1, t2, H1, H2, o, v)

    print("    ==> DEA-EOMCC(2p) iterations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dR|     Wall Time     Memory")
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
        R = np.dot(B[:, :curr_size], alpha)
        restart_block[:, niter % nrest] = R

        # calculate residual vector
        residual = np.dot(sigma[:, :curr_size], alpha) - omega * R
        res_norm = np.linalg.norm(residual)
        delta_e = omega - omega_old

        if res_norm < convergence and abs(delta_e) < convergence:
            toc = time.time()
            minutes, seconds = divmod(toc - tic, 60)
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
            break

        # update residual vector
        q = update(residual.reshape(nunocc, nunocc),
                   omega,
                   e_ab)

        for p in range(curr_size):
            b = B[:, p] / np.linalg.norm(B[:, p])
            q -= np.dot(b.T, q) * b
        q *= 1.0 / np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[:, curr_size] = q
            sigma[:, curr_size] = HR(q.reshape(nunocc, nunocc),
                                     t1, t2, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[:, j] = restart_block[:, j]
                sigma[:, j] = HR(restart_block[:, j].reshape(nunocc, nunocc),
                                 t1, t2, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        print("DEA-EOMCC(2p) iterations did not converge")

    # Save the final converged root in an excitation tuple
    R = (R.reshape(nunocc, nunocc))
    # r0 for a root in DEA is 0 by definition
    r0 = 0.0
    # Compute relative excitation level diagnostic
    #rel = calc_rel_dea(R[0], R[1])
    rel = 0.0
    return R, omega, r0, rel

def update(r1, omega, e_ab):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""

    for a in range(r1.shape[0]):
        for b in range(r1.shape[1]):
            denom = omega - e_ab[a, b]
            if denom == 0: continue
            r1[a, b] /= denom

    return r1.flatten()


def HR(r1, t1, t2, H1, H2, o, v):
    """Compute the matrix-vector product H * R, where
    H is the CCSD similarity-transformed Hamiltonian and R is
    the EA-EOMCC linear excitation operator."""

    # update R1
    HR1 = build_HR1(r1, H1, H2, o, v)

    return HR1.flatten()


def build_HR1(r1, H1, H2, o, v):
    """Compute the projection of HR on 2p excitations
        X[a, b] = < ab | [ HBar(CCSD) * (R1 + R2) ]_C | 0 >
    """
    X1 = np.einsum("ae,eb->ab", H1[v, v], r1, optimize=True)
    X1 += 0.25 * np.einsum("abef,ef->ab", H2[v, v, v, v], r1, optimize=True)
    # antisymmetrize A(ab)
    X1 -= np.transpose(X1, (1, 0))
    return X1
