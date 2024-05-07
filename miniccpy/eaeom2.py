import time
import numpy as np
from miniccpy.utilities import get_memory_usage

def kernel(R0, T, omega, H1, H2, o, v, maxit=80, convergence=1.0e-07, max_size=20, nrest=1):
    """
    Diagonalize the similarity-transformed CCSD Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
    from miniccpy.energy import calc_rel_ea

    eps = np.diagonal(H1)
    n = np.newaxis
    e_abj = (eps[v, n, n] + eps[n, v, n] - eps[n, n, o])
    e_a = eps[v]

    t1, t2 = T

    nunocc, nocc = t1.shape
    n1 = nunocc
    n2 = nocc * nunocc**2
    ndim = n1 + n2
    
    if len(R0) < ndim:
        R = np.zeros(ndim)
        R[:len(R0)] = R0

    # Allocate the B and sigma matrices
    sigma = np.zeros((ndim, max_size))
    B = np.zeros((ndim, max_size))
    restart_block = np.zeros((ndim, nrest))

    # Initial values
    B[:, 0] = R
    sigma[:, 0] = HR(R[:n1].reshape(nunocc),
                     R[n1:].reshape(nunocc, nunocc, nocc),
                     t1, t2, H1, H2, o, v)

    print("    ==> EA-EOMCC(2p-1h) iterations <==")
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
        q = update(residual[:n1].reshape(nunocc),
                   residual[n1:].reshape(nunocc, nunocc, nocc),
                   omega,
                   e_a,
                   e_abj)
        for p in range(curr_size):
            b = B[:, p] / np.linalg.norm(B[:, p])
            q -= np.dot(b.T, q) * b
        q *= 1.0 / np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[:, curr_size] = q
            sigma[:, curr_size] = HR(q[:n1].reshape(nunocc),
                                     q[n1:].reshape(nunocc, nunocc, nocc),
                                     t1, t2, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[:, j] = restart_block[:, j]
                sigma[:, j] = HR(restart_block[:n1, j].reshape(nunocc),
                                 restart_block[n1:, j].reshape(nunocc, nunocc, nocc),
                                 t1, t2, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        print("EA-EOMCC(2p-1h) iterations did not converge")

    # Save the final converged root in an excitation tuple
    R = (R[:n1].reshape(nunocc), R[n1:].reshape(nunocc, nunocc, nocc))
    # Set the r0 to 0
    r0 = 0.0
    # Compute the REL metric
    rel = calc_rel_ea(R[0], R[1])
    return R, omega, r0, rel

def update(r1, r2, omega, e_a, e_abj):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""

    for a, d_a in enumerate(e_a):
        denom = omega - d_a
        if denom == 0: continue
        r1[a] /= denom
    r2 /= (omega - e_abj)

    return np.hstack([r1.flatten(), r2.flatten()])


def HR(r1, r2, t1, t2, H1, H2, o, v):
    """Compute the matrix-vector product H * R, where
    H is the CCSD similarity-transformed Hamiltonian and R is
    the EA-EOMCC linear excitation operator."""

    # update R1
    HR1 = build_HR1(r1, r2, H1, H2, o, v)
    # update R2
    HR2 = build_HR2(r1, r2, t1, t2, H1, H2, o, v)

    return np.hstack( [HR1.flatten(), HR2.flatten()] )


def build_HR1(r1, r2, H1, H2, o, v):
    """Compute the projection of HR on 1p excitations
        X[a] = < a | [ HBar(CCSD) * (R1 + R2) ]_C | 0 >
    """
    X1 = np.einsum("ae,e->a", H1[v, v], r1, optimize=True)
    X1 += 0.5 * np.einsum("anef,efn->a", H2[v, o, v, v], r2, optimize=True)
    X1 += np.einsum("me,aem->a", H1[o, v], r2, optimize=True)
    return X1


def build_HR2(r1, r2, t1, t2, H1, H2, o, v):
    """Compute the projection of HR on 2p-1h excitations
        X[a, b, j] = < jab | [ HBar(CCSD) * (R1 + R2) ]_C | 0 >
    """
    X2 = 0.5 * np.einsum("baje,e->abj", H2[v, v, o, v], r1, optimize=True)
    X2 -= 0.5 * np.einsum("mj,abm->abj", H1[o, o], r2, optimize=True)
    X2 += 0.25 * np.einsum("abef,efj->abj", H2[v, v, v, v], r2, optimize=True)
    I1 = 0.5 * np.einsum("mnef,efn->m", H2[o, o, v, v], r2, optimize=True)
    X2 -= 0.5 * np.einsum("m,abmj->abj", I1, t2, optimize=True)
    X2 += np.einsum("ae,ebj->abj", H1[v, v], r2, optimize=True)
    X2 += np.einsum("bmje,aem->abj", H2[v, o, o, v], r2, optimize=True)
    X2 -= np.transpose(X2, (1, 0, 2))
    return X2

