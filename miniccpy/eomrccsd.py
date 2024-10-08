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
    from miniccpy.energy import calc_r0_rhf, calc_rel_rhf

    remove_file("eomcc-vectors.hdf5")
    if out_of_core:
        f = h5py.File("eomcc-vectors.hdf5", "w")

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
    if len(R0) < ndim:
        R = np.zeros(ndim)
        R[:len(R0)] = R0
    else:
        R = R0.copy()

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
    sigma[0, :] = HR(R[:n1].reshape(nunocc, nocc),
                     R[n1:].reshape(nunocc, nunocc, nocc, nocc),
                     t1, t2, H1, H2, o, v)

    print("    ==> R-EOMCCSD iterations <==")
    print("    The initial guess energy = ", omega)
    print("")
    print("     Iter               Energy                 |dE|                 |dR|     Wall Time     Memory")
    curr_size = 1
    for niter in range(maxit):
        tic = time.time()
        # store old energy
        omega_old = omega

        # solve projection subspace eigenproblem: G_{IJ} = sum_K B_{KI} S_{KJ} (vectorized)
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
        q = update(residual[:n1].reshape(nunocc, nocc),
                   residual[n1:].reshape(nunocc, nunocc, nocc, nocc),
                   omega,
                   e_ai,
                   e_abij)
        for p in range(curr_size):
            b = B[p, :] / np.linalg.norm(B[p, :])
            q -= np.dot(b.T, q) * b
        q *= 1.0 / np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[curr_size, :] = q
            sigma[curr_size, :] = HR(q[:n1].reshape(nunocc, nocc),
                                     q[n1:].reshape(nunocc, nunocc, nocc, nocc),
                                     t1, t2, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[j, :] = restart_block[:, j]
                sigma[j, :] = HR(restart_block[:n1, j].reshape(nunocc, nocc),
                                 restart_block[n1:, j].reshape(nunocc, nunocc, nocc, nocc),
                                 t1, t2, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        print("EOMCCSD iterations did not converge")

    # Save the final converged root in an excitation tuple
    R = (R[:n1].reshape(nunocc, nocc), R[n1:].reshape(nunocc, nunocc, nocc, nocc))
    # Calculate r0 for the root
    r0 = calc_r0_rhf(R[0], R[1], H1, H2, omega, o, v)
    # Compute relative excitation level diagnostic
    rel = calc_rel_rhf(r0, R[0], R[1])
    # remove the HDF5 file
    remove_file("eomcc-vectors.hdf5")
    return R, omega, r0, rel

def update(r1, r2, omega, e_ai, e_abij):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""

    r1 /= (omega - e_ai + 1.0e-012)
    r2 /= (omega - e_abij + 1.0e-012)

    return np.hstack([r1.flatten(), r2.flatten()])

def HR(r1, r2, t1, t2, H1, H2, o, v):
    """Compute the matrix-vector product H * R, where
    H is the CCSD similarity-transformed Hamiltonian and R is
    the EOMCCSD linear excitation operator."""

    # update R1
    HR1 = build_HR1(r1, r2, H1, H2, o, v)
    # update R2
    HR2 = build_HR2(r1, r2, t1, t2, H1, H2, o, v)

    return np.hstack( [HR1.flatten(), HR2.flatten()] )

def build_HR1(r1, r2, H1, H2, o, v):
    """Compute the projection of HR on singles
        X[a, i] = < ia | [ HBar(CCSD) * (R1 + R2) ]_C | 0 >
    """
    X1 = -np.einsum("mi,am->ai", H1[o, o], r1, optimize=True)
    X1 += np.einsum("ae,ei->ai", H1[v, v], r1, optimize=True)
    X1 += 2.0 * np.einsum("me,aeim->ai", H1[o, v], r2, optimize=True)
    X1 -= np.einsum("me,aemi->ai", H1[o, v], r2, optimize=True)
    X1 += 2.0 * np.einsum("amie,em->ai", H2[v, o, o, v], r1, optimize=True)
    X1 -= np.einsum("amei,em->ai", H2[v, o, v, o], r1, optimize=True)
    X1 -= 2.0 * np.einsum("mnif,afmn->ai", H2[o, o, o, v], r2, optimize=True)
    X1 += np.einsum("nmif,afmn->ai", H2[o, o, o, v], r2, optimize=True)
    X1 += 2.0 * np.einsum("anef,efin->ai", H2[v, o, v, v], r2, optimize=True)
    X1 -= np.einsum("anfe,efin->ai", H2[v, o, v, v], r2, optimize=True)
    return X1

def build_HR2(r1, r2, t1, t2, H1, H2, o, v):
    """Compute the projection of HR on doubles
        X[a, b, i, j] = < ijab | [ HBar(CCSD) * (R1 + R2) ]_C | 0 >
    """
    # intermediates
    X_oo = (
            + 2.0 * np.einsum("mnjf,fn->mj", H2[o, o, o, v], r1, optimize=True)
            - np.einsum("nmjf,fn->mj", H2[o, o, o, v], r1, optimize=True)
            + 2.0 * np.einsum("mnef,efjn->mj", H2[o, o, v, v], r2, optimize=True)
            - np.einsum("nmef,efjn->mj", H2[o, o, v, v], r2, optimize=True)
    )
    X_vv = (
            + 2.0 * np.einsum("bnef,fn->be", H2[v, o, v, v], r1, optimize=True)
            - np.einsum("bnfe,fn->be", H2[v, o, v, v], r1, optimize=True)
            - 2.0 * np.einsum("mnef,bfmn->be", H2[o, o, v, v], r2, optimize=True)
            + np.einsum("nmef,bfmn->be", H2[o, o, v, v], r2, optimize=True)
    )
    # < IJAB | (H(2)*(R1+R2))_C | 0 >
    X2 = np.einsum("ae,ebij->abij", H1[v, v], r2, optimize=True)
    X2 -= np.einsum("mi,abmj->abij", H1[o, o], r2, optimize=True)
    X2 += 0.5 * np.einsum("mnij,abmn->abij", H2[o, o, o, o], r2, optimize=True)
    X2 += 0.5 * np.einsum("abef,efij->abij", H2[v, v, v, v], r2, optimize=True)
    X2 += np.einsum("baje,ei->abij", H2[v, v, o, v], r1, optimize=True)
    X2 -= np.einsum("bmji,am->abij", H2[v, o, o, o], r1, optimize=True)
    X2 += np.einsum("ae,ebij->abij", X_vv, t2, optimize=True)
    X2 -= np.einsum("mi,abmj->abij", X_oo, t2, optimize=True)
    X2 += 2.0 * np.einsum("amie,ebmj->abij", H2[v, o, o, v], r2, optimize=True)
    X2 -= np.einsum("amie,ebjm->abij", H2[v, o, o, v], r2, optimize=True)
    X2 -= np.einsum("amei,ebmj->abij", H2[v, o, v, o], r2, optimize=True)
    X2 -= np.einsum("amej,ebim->abij", H2[v, o, v, o], r2, optimize=True)
    X2 += X2.transpose(1, 0, 3, 2)
    return X2

