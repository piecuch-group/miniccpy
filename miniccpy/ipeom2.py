import time
import numpy as np

def kernel(R0, T, omega, H1, H2, o, v, maxit=80, convergence=1.0e-07, max_size=20, nrest=1):
    """
    Diagonalize the similarity-transformed CCSD Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
    from miniccpy.energy import calc_rel_ip

    eps = np.diagonal(H1)
    n = np.newaxis
    e_ibj = (-eps[o, n, n] + eps[n, v, n] - eps[n, n, o])
    e_i = -eps[o]

    t1, t2 = T

    nunocc, nocc = t1.shape
    n1 = nocc
    n2 = nocc**2 * nunocc
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
    sigma[:, 0] = HR(R[:n1].reshape(nocc),
                     R[n1:].reshape(nocc, nunocc, nocc),
                     t1, t2, H1, H2, o, v)

    print("    ==> IP-EOMCC(2h-1p) iterations <==")
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
        q = update(residual[:n1].reshape(nocc),
                   residual[n1:].reshape(nocc, nunocc, nocc),
                   omega,
                   e_i,
                   e_ibj)
        for p in range(curr_size):
            b = B[:, p] / np.linalg.norm(B[:, p])
            q -= np.dot(b.T, q) * b
        q *= 1.0 / np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[:, curr_size] = q
            sigma[:, curr_size] = HR(q[:n1].reshape(nocc),
                                     q[n1:].reshape(nocc, nunocc, nocc),
                                     t1, t2, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[:, j] = restart_block[:, j]
                sigma[:, j] = HR(restart_block[:n1, j].reshape(nocc),
                                 restart_block[n1:, j].reshape(nocc, nunocc, nocc),
                                 t1, t2, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1
    else:
        print("IP-EOMCC(2h-1p) iterations did not converge")

    # Set the r0 and rel trivially to 0
    r0 = 0.0
    rel = calc_rel_ip(R[:n1].reshape(nocc),
                      R[n1:].reshape(nocc, nunocc, nocc))
    return R, omega, r0, rel

def update(r1, r2, omega, e_i, e_ibj):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""

    for i, d_i in enumerate(e_i):
        denom = omega - d_i
        if denom == 0: continue
        r1[i] /= denom
    r2 /= (omega - e_ibj)

    return np.hstack([r1.flatten(), r2.flatten()])


def HR(r1, r2, t1, t2, H1, H2, o, v):
    """Compute the matrix-vector product H * R, where
    H is the CCSD similarity-transformed Hamiltonian and R is
    the IP-EOMCC linear excitation operator."""

    # update R1
    HR1 = build_HR1(r1, r2, H1, H2, o, v)
    # update R2
    HR2 = build_HR2(r1, r2, t1, t2, H1, H2, o, v)

    return np.hstack( [HR1.flatten(), HR2.flatten()] )


def build_HR1(r1, r2, H1, H2, o, v):
    """Compute the projection of HR on 1h excitations
        X[i] = < i | [ HBar(CCSD) * (R1 + R2) ]_C | 0 >
    """
    X1 = -np.einsum("mi,m->i", H1[o, o], r1, optimize=True)
    X1 -= 0.5 * np.einsum("mnif,mfn->i", H2[o, o, o, v], r2, optimize=True)
    X1 += np.einsum("me,iem->i", H1[o, v], r2, optimize=True)
    return X1


def build_HR2(r1, r2, t1, t2, H1, H2, o, v):
    """Compute the projection of HR on 2h-1p excitations
        X[i, b, j] = < ijb | [ HBar(CCSD) * (R1 + R2) ]_C | 0 >
    """
    I1_v = -0.5 * np.einsum("mnef,mfn->e", H2[o, o, v, v], r2, optimize=True)

    X2 = -0.5 * np.einsum("bmji,m->ibj", H2[v, o, o, o], r1, optimize=True)
    X2 += 0.5 * np.einsum("be,iej->ibj", H1[v, v], r2, optimize=True)
    X2 += 0.25 * np.einsum("mnij,mbn->ibj", H2[o, o, o, o], r2, optimize=True)
    X2 += 0.5 * np.einsum("e,ebij->ibj", I1_v, t2, optimize=True)
    X2 -= np.einsum("mi,mbj->ibj", H1[o, o], r2, optimize=True)
    X2 += np.einsum("bmje,iem->ibj", H2[v, o, o, v], r2, optimize=True)
    X2 -= np.transpose(X2, (2, 1, 0))
    return X2

