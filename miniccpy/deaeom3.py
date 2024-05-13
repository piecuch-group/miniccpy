import time
import numpy as np
import h5py
from miniccpy.utilities import get_memory_usage, remove_file

def kernel(R0, T, omega, H1, H2, o, v, maxit=80, convergence=1.0e-07, max_size=20, nrest=1, out_of_core=False):
    """
    Diagonalize the similarity-transformed CCSD Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
    from miniccpy.energy import calc_rel_dea

    remove_file("eomcc-vectors.hdf5")
    if out_of_core:
        f = h5py.File("eomcc-vectors.hdf5", "w")

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
    if out_of_core:
        sigma = f.create_dataset("sigma", (max_size, ndim), dtype=np.float64)
        B = f.create_dataset("bmatrix", (max_size, ndim), dtype=np.float64)
    else:
        sigma = np.zeros((max_size, ndim))
        B = np.zeros((max_size, ndim))
    restart_block = np.zeros((ndim, nrest))
    G = np.zeros((max_size, max_size))

    # Initial values
    B[0, :] = R
    sigma[0, :] = HR(R[:n1].reshape(nunocc, nunocc),
                     R[n1:].reshape(nunocc, nunocc, nunocc, nocc),
                     t1, t2, H1, H2, o, v)

    print("    ==> DEA-EOMCC(3p-1h) iterations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dR|     Wall Time     Memory")
    curr_size = 1
    for niter in range(maxit):
        tic = time.time()
        # store old energy
        omega_old = omega

        # solve projection subspace eigenproblem
        G[curr_size - 1, :curr_size] = np.einsum("k,pk->p", B[curr_size - 1, :], sigma[:curr_size, :])
        G[:curr_size, curr_size - 1] = np.einsum("k,pk->p", sigma[curr_size - 1, :], B[:curr_size, :])
        e, alpha_full = np.linalg.eig(G[:curr_size, :curr_size])

        # select root based on maximum overlap with initial guess
        idx = np.argsort(abs(alpha_full[0, :]))
        iselect = idx[-1]

        alpha = np.real(alpha_full[:, iselect])

        # Get the eigenpair of interest
        omega = np.real(e[iselect])
        R = np.dot(B[:curr_size, :].T, alpha)
        restart_block[:, niter % nrest] = R

        # calculate residual vector
        residual = np.dot(sigma[:curr_size, :].T, alpha) - omega * R
        res_norm = np.linalg.norm(residual)
        delta_e = omega - omega_old

        if res_norm < convergence and abs(delta_e) < convergence:
            toc = time.time()
            minutes, seconds = divmod(toc - tic, 60)
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
            break

        # update residual vector
        q = update(residual[:n1].reshape(nunocc, nunocc),
                   residual[n1:].reshape(nunocc, nunocc, nunocc, nocc),
                   omega,
                   e_ab,
                   e_abck)
        for p in range(curr_size):
            b = B[p, :] / np.linalg.norm(B[p, :])
            q -= np.dot(b.T, q) * b
        q *= 1.0 / np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[curr_size, :] = q
            sigma[curr_size, :] = HR(q[:n1].reshape(nunocc, nunocc),
                                     q[n1:].reshape(nunocc, nunocc, nunocc, nocc),
                                     t1, t2, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[j, :] = restart_block[:, j]
                sigma[j, :] = HR(restart_block[:n1, j].reshape(nunocc, nunocc),
                                 restart_block[n1:, j].reshape(nunocc, nunocc, nunocc, nocc),
                                 t1, t2, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        print("DEA-EOMCC(3p-1h) iterations did not converge")

    # Save the final converged root in an excitation tuple
    R = (R[:n1].reshape(nunocc, nunocc), R[n1:].reshape(nunocc, nunocc, nunocc, nocc))
    # r0 for a root in DEA is 0 by definition
    r0 = 0.0
    # Compute relative excitation level diagnostic
    rel = calc_rel_dea(R[0], R[1])
    # remove the HDF5 file
    remove_file("eomcc-vectors.hdf5")
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
