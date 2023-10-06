import time
import numpy as np

def kernel(R0, T, omega, H1, H2, o, v, maxit=80, convergence=1.0e-07, max_size=20, nrest=1):
    """
    Diagonalize the similarity-transformed CCSD Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
    from miniccpy.energy import calc_rel_dea

    eps = np.diagonal(H1)
    n = np.newaxis
    e_abck = (eps[v, n, n, n] + eps[n, v, n, n] + eps[n, n, v, n] - eps[n, n, n, o])
    e_ab = (eps[v, n] + eps[n, v])

    t1, t2 = T

    nunocc, nocc = t1.shape
    n1 = nunocc**2
    n2 = nunocc**3 * nocc
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
    sigma[:, 0] = HR(R[:n1].reshape(nunocc, nunocc),
                     R[n1:].reshape(nunocc, nunocc, nunocc, nocc),
                     t1, t2, H1, H2, o, v)

    print("    ==> DEA-EOMCC(3p-1h) iterations <==")
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
        R = np.dot(B[:, :curr_size], alpha)
        restart_block[:, niter % nrest] = R

        # calculate residual vector
        residual = np.dot(sigma[:, :curr_size], alpha) - omega * R
        res_norm = np.linalg.norm(residual)
        delta_e = omega - omega_old

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(niter, omega, delta_e, res_norm, minutes, seconds))
        if res_norm < convergence and abs(delta_e) < convergence:
            break

        # update residual vector
        q = update(residual[:n1].reshape(nunocc, nunocc),
                   residual[n1:].reshape(nunocc, nunocc, nunocc, nocc),
                   omega,
                   e_ab,
                   e_abck)
        for p in range(curr_size):
            b = B[:, p] / np.linalg.norm(B[:, p])
            q -= np.dot(b.T, q) * b
        q *= 1.0 / np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[:, curr_size] = q
            sigma[:, curr_size] = HR(q[:n1].reshape(nunocc, nunocc),
                                     q[n1:].reshape(nunocc, nunocc, nunocc, nocc),
                                     t1, t2, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[:, j] = restart_block[:, j]
                sigma[:, j] = HR(restart_block[:n1, j].reshape(nunocc, nunocc),
                                 restart_block[n1:, j].reshape(nunocc, nunocc, nunocc, nocc),
                                 t1, t2, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1
    else:
        print("DEA-EOMCC(3p-1h) iterations did not converge")

    # Save the final converged root in an excitation tuple
    R = (R[:n1].reshape(nunocc, nunocc), R[n1:].reshape(nunocc, nunocc, nunocc, nocc))
    # r0 for a root in DEA is 0 by definition
    r0 = 0.0
    # Compute relative excitation level diagnostic
    rel = calc_rel_dea(R[0], R[1])
    return R, omega, r0, rel

def update(r1, r2, omega, e_ab, e_abck):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""

    for a in range(r1.shape[0]):
        for b in range(r1.shape[1]):
            denom = omega - e_ab[a, b]
            if denom == 0: continue
            r1[a, b] /= denom
    #r1 /= (omega - e_ab)
    r2 /= (omega - e_abck)

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
    """Compute the projection of HR on 2p excitations
        X[a, b] = < ab | [ HBar(CCSD) * (R1 + R2) ]_C | 0 >
    """
    X1 = np.einsum("ae,eb->ab", H1[v, v], r1, optimize=True)
    X1 += 0.25 * np.einsum("abef,ef->ab", H2[v, v, v, v], r1, optimize=True)
    X1 += 0.5 * np.einsum("me,abem->ab", H1[o, v], r2, optimize=True)
    X1 += 0.5 * np.einsum("anef,ebfn->ab", H2[v, o, v, v], r2, optimize=True)
    # antisymmetrize A(ab)
    X1 -= np.transpose(X1, (1, 0))
    return X1


def build_HR2(r1, r2, t1, t2, H1, H2, o, v):
    """Compute the projection of HR on 3p-1h excitations
        X[a, b, c, k] = < kabc | [ HBar(CCSD) * (R1 + R2) ]_C | 0 >
    """
    I_vo = (
            0.5 * np.einsum("amef,ef->am", H2[v, o, v, v], r1, optimize=True)
            + 0.5 * np.einsum("mnef,bfem->bn", H2[o, o, v, v], r2, optimize=True)
    )

    X2 = -(3.0 / 6.0) * np.einsum("am,bcmk->abck", I_vo, t2, optimize=True)
    X2 += (3.0 / 6.0) * np.einsum("cbke,ae->abck", H2[v, v, o, v], r1, optimize=True)
    X2 -= (1.0 / 6.0) * np.einsum("mk,abcm->abck", H1[o, o], r2, optimize=True)
    X2 += (3.0 / 6.0) * np.einsum("be,aeck->abck", H1[v, v], r2, optimize=True)
    X2 += (3.0 / 12.0) * np.einsum("abef,efck->abck", H2[v, v, v, v], r2, optimize=True)
    X2 += (3.0 / 6.0) * np.einsum("cmke,abem->abck", H2[v, o, o, v], r2, optimize=True)
    # antisymmetrize A(abc)
    X2 -= np.transpose(X2, (0, 2, 1, 3)) # A(bc)
    X2 -= np.transpose(X2, (1, 0, 2, 3)) + np.transpose(X2, (2, 1, 0, 3)) # A(a/bc)
    return X2
