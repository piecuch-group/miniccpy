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
    from miniccpy.energy import calc_rel_dip

    remove_file("eomcc-vectors.hdf5")
    if out_of_core:
        f = h5py.File("eomcc-vectors.hdf5", "w")

    eps = np.diagonal(H1)
    n = np.newaxis
    e_ijcdkl = (-eps[o, n, n, n, n, n] - eps[n, o, n, n, n, n] + eps[n, n, v, n, n, n] + eps[n, n, n, v, n, n] - eps[n, n, n, n, o, n] - eps[n, n, n, n, n, o])
    e_ijck = (-eps[o, n, n, n] - eps[n, o, n, n] + eps[n, n, v, n] - eps[n, n, n, o])
    e_ij = (-eps[o, n] - eps[n, o])

    t1, t2, t3 = T

    nunocc, nocc = t1.shape
    n1 = nocc**2
    n2 = nocc**3 * nunocc
    n3 = nocc**4 * nunocc**2
    ndim = n1 + n2 + n3
    
    if len(R0) <= ndim:
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
    sigma[0, :] = HR(R[:n1].reshape(nocc, nocc),
                     R[n1:n1+n2].reshape(nocc, nocc, nunocc, nocc),
                     R[n1+n2:].reshape(nocc, nocc, nunocc, nunocc, nocc, nocc),
                     t1, t2, t3, H1, H2, o, v)

    print("    ==> DIP-EOMCCSDT iterations <==")
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
        q = update(residual[:n1].reshape(nocc, nocc),
                   residual[n1:n1+n2].reshape(nocc, nocc, nunocc, nocc),
                   residual[n1+n2:].reshape(nocc, nocc, nunocc, nunocc, nocc, nocc),
                   omega,
                   e_ij,
                   e_ijck,
                   e_ijcdkl)
        for p in range(curr_size):
            b = B[p, :] / np.linalg.norm(B[p, :])
            q -= np.dot(b.T, q) * b
        q *= 1.0 / np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[curr_size, :] = q
            sigma[curr_size, :] = HR(q[:n1].reshape(nocc, nocc),
                                     q[n1:n1+n2].reshape(nocc, nocc, nunocc, nocc),
                                     q[n1+n2:].reshape(nocc, nocc, nunocc, nunocc, nocc, nocc),
                                     t1, t2, t3, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[j, :] = restart_block[:, j]
                sigma[j, :] = HR(restart_block[:n1, j].reshape(nocc, nocc),
                                 restart_block[n1:n1+n2, j].reshape(nocc, nocc, nunocc, nocc),
                                 restart_block[n1+n2:, j].reshape(nocc, nocc, nunocc, nunocc, nocc, nocc),
                                 t1, t2, t3, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        print("DIP-EOMCCSDT iterations did not converge")

    # Save the final converged root in an excitation tuple
    R = (R[:n1].reshape(nocc, nocc), R[n1:n1+n2].reshape(nocc, nocc, nunocc, nocc), R[n1+n2:].reshape(nocc, nocc, nunocc, nunocc, nocc, nocc))
    # r0 for a root in DIP is 0 by definition
    r0 = 0.0
    # Compute relative excitation level diagnostic
    rel = calc_rel_dip(R[0], R[1])
    # remove the HDF5 file
    remove_file("eomcc-vectors.hdf5")
    return R, omega, r0, rel

def update(r1, r2, r3, omega, e_ij, e_ijck, e_ijcdkl):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""

    r1 /= (omega - e_ij)
    r2 /= (omega - e_ijck)
    r3 /= (omega - e_ijcdkl)

    return np.hstack([r1.flatten(), r2.flatten(), r3.flatten()])

def HR(r1, r2, r3, t1, t2, t3, H1, H2, o, v):
    """Compute the matrix-vector product H * R, where
    H is the CCSD similarity-transformed Hamiltonian and R is
    the DIP-EOMCC linear excitation operator."""

    # update R1
    HR1 = build_HR1(r1, r2, r3, H1, H2, o, v)
    # update R2
    HR2 = build_HR2(r1, r2, r3, t1, t2, t3, H1, H2, o, v)
    # update R3
    HR3 = build_HR3(r1, r2, r3, t1, t2, t3, H1, H2, o, v)

    return np.hstack( [HR1.flatten(), HR2.flatten(), HR3.flatten()] )

def build_HR1(r1, r2, r3, H1, H2, o, v):
    """Compute the projection of HR on 2h excitations
        X[i, j] = < ij | [ HBar(CCSDT) * (R1 + R2 + R3) ]_C | 0 >
    """
    X1 = -np.einsum("mi,mj->ij", H1[o, o], r1, optimize=True)
    X1 += 0.25 * np.einsum("mnij,mn->ij", H2[o, o, o, o], r1, optimize=True)
    X1 += 0.5 * np.einsum("me,ijem->ij", H1[o, v], r2, optimize=True)
    X1 -= 0.5 * np.einsum("mnif,mjfn->ij", H2[o, o, o, v], r2, optimize=True)
    # terms contracted with R(4h-2p)
    X1 += 0.125 * np.einsum("mnef,ijefmn->ij", H2[o, o, v, v], r3, optimize=True)
    # antisymmetrize A(ij)
    X1 -= np.transpose(X1, (1, 0))
    return X1

def build_HR2(r1, r2, r3, t1, t2, t3, H1, H2, o, v):
    """Compute the projection of HR on 3h-1p excitations
        X[i, j, c, k] = < ijkc | [ HBar(CCSDT) * (R1 + R2 + R3) ]_C | 0 >
    """
    I_vo = (
            0.5 * np.einsum("mnie,mn->ie", H2[o, o, o, v], r1, optimize=True)
            - 0.5 * np.einsum("mnef,jnem->jf", H2[o, o, v, v], r2, optimize=True)
    )
    I_vv = (
            0.5 * np.einsum("mnef,mn->ef", H2[o, o, v, v], r1, optimize=True)
    )
    X2 = (3.0 / 6.0) * np.einsum("ie,ecjk->ijck", I_vo, t2, optimize=True)
    X2 -= (3.0 / 6.0) * np.einsum("cmki,mj->ijck", H2[v, o, o, o], r1, optimize=True)
    X2 += (1.0 / 6.0) * np.einsum("ce,ijek->ijck", H1[v, v], r2, optimize=True)
    X2 -= (3.0 / 6.0) * np.einsum("mk,ijcm->ijck", H1[o, o], r2, optimize=True)
    X2 += (3.0 / 12.0) * np.einsum("mnij,mnck->ijck", H2[o, o, o, o], r2, optimize=True)
    X2 += (3.0 / 6.0) * np.einsum("cmke,ijem->ijck", H2[v, o, o, v], r2, optimize=True)
    # terms contracted with T3
    X2 += (1.0 / 12.0) * np.einsum("ef,efcijk->ijck", I_vv, t3, optimize=True)
    # terms contracted with R(4h-2p)
    X2 += (1.0 / 6.0) * np.einsum("me,ijcekm->ijck", H1[o, v], r3, optimize=True)
    X2 -= (3.0 / 12.0) * np.einsum("mnkf,ijcfmn->ijck", H2[o, o, o, v], r3, optimize=True)
    X2 += (1.0 / 12.0) * np.einsum("cnef,ijefkn->ijck", H2[v, o, v, v], r3, optimize=True)
    # antisymmetrize A(ijk)
    X2 -= np.transpose(X2, (0, 3, 2, 1)) # A(jk)
    X2 -= np.transpose(X2, (1, 0, 2, 3)) + np.transpose(X2, (3, 1, 2, 0)) # A(i/jk)
    return X2

def build_HR3(r1, r2, r3, t1, t2, t3, H1, H2, o, v):
    """Compute the projection of HR on 4h-2p excitations
        X[i, j, c, d, k, l] = < ijklcd | [ HBar(CCSDT) * (R1 + R2 + R3) ]_C | 0 >
    """
    # Intermediates
    I_ov = -np.einsum("me,im->ie", H1[o, v], r1, optimize=True)
    I_vv = (
            0.5 * np.einsum("mnef,mn->ef", H2[o, o, v, v], r1, optimize=True)
    )
    # I(ijmk)
    I_oooo = (
          (3.0 / 6.0) * np.einsum("nmke,ijem->ijnk", H2[o, o, o, v], r2, optimize=True)
        - (3.0 / 6.0) * np.einsum("mnik,mj->ijnk", H2[o, o, o, o], r1, optimize=True) # T2
        + (1.0 / 12.0) * np.einsum("mnef,ijefkn->ijmk", H2[o, o, v, v], r3, optimize=True)
    )
    # antisymmetrize A(ijk)
    I_oooo -= np.transpose(I_oooo, (0, 3, 2, 1)) # A(jk)
    I_oooo -= np.transpose(I_oooo, (1, 0, 2, 3)) + np.transpose(I_oooo, (3, 1, 2, 0)) # A(i/jk)
    # I(ijce)
    I_oovv = (
        (1.0 / 2.0) * np.einsum("cmfe,ijem->ijcf", H2[v, o, v, v], r2, optimize=True)
        + np.einsum("bmje,mk->jkbe", H2[v, o, o, v], r1, optimize=True) # T2
        + 0.5 * np.einsum("nmie,njcm->ijce", H2[o, o, o, v], r2, optimize=True)
        - (1.0 / 4.0) * np.einsum("mnef,ijcfmn->ijce", H2[o, o, v, v], r3, optimize=True)
        + 0.25 * np.einsum("ef,edil->lidf", I_vv, t2, optimize=True)
    )
    # antisymmetrize A(ij)
    I_oovv -= np.transpose(I_oovv, (1, 0, 2, 3))
    # I(ijem)
    I_oovo = (
        -np.einsum("nmje,in->ijem", H2[o, o, o, v], r1, optimize=True)
        + 0.5 * np.einsum("mnef,ijfn->ijem", H2[o, o, v, v], r2, optimize=True)
    )
    I_oovo -= np.transpose(I_oovo, (1, 0, 2, 3))
    #
    # Moment-like terms
    X3 = (4.0 / 48.0) * np.einsum("dcle,ijek->ijcdkl", H2[v, v, o, v], r2, optimize=True) # T2
    X3 -= (12.0 / 48.0) * np.einsum("dmlk,ijcm->ijcdkl", H2[v, o, o, o], r2, optimize=True) # T2
    X3 -= (4.0 / 48.0) * np.einsum("ijmk,cdml->ijcdkl", I_oooo, t2, optimize=True)
    X3 += (12.0 / 48.0) * np.einsum("ijce,edkl->ijcdkl", I_oovv, t2, optimize=True)
    # terms contracted with T3
    X3 += (4.0 / 48.0) * np.einsum("ie,ecdjkl->ijcdkl", I_ov, t3, optimize=True)
    X3 += (6.0 / 48.0) * np.einsum("ijem,cdeklm->ijcdkl", I_oovo, t3, optimize=True)
    # terms contracted with R3
    X3 += (2.0 / 48.0) * np.einsum("de,ijcekl->ijcdkl", H1[v, v], r3, optimize=True) # T2
    X3 -= (4.0 / 48.0) * np.einsum("mi,mjcdkl->ijcdkl", H1[o, o], r3, optimize=True) # T2
    X3 += (1.0 / 96.0) * np.einsum("cdef,ijefkl->ijcdkl", H2[v, v, v, v], r3, optimize=True)
    X3 += (6.0 / 96.0) * np.einsum("mnij,mncdkl->ijcdkl", H2[o, o, o, o], r3, optimize=True)
    X3 += (8.0 / 48.0) * np.einsum("dmle,ijcekm->ijcdkl", H2[v, o, o, v], r3, optimize=True)
    # antisymmetrize A(ijkl)A(cd)
    X3 -= np.transpose(X3, (0, 1, 3, 2, 4, 5)) # A(cd)
    X3 -= np.transpose(X3, (0, 4, 2, 3, 1, 5)) # A(jk)
    X3 -= np.transpose(X3, (1, 0, 2, 3, 4, 5)) + np.transpose(X3, (4, 1, 2, 3, 0, 5)) # A(i/jk)
    X3 -= np.transpose(X3, (5, 1, 2, 3, 4, 0)) + np.transpose(X3, (0, 5, 2, 3, 4, 1)) + np.transpose(X3, (0, 1, 2, 3, 5, 4)) # A(l/ijk)
    return X3

