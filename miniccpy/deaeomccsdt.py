import time
import numpy as np
import h5py
from miniccpy.utilities import get_memory_usage, remove_file

def kernel(R0, T, omega, H1, H2, o, v, maxit=80, convergence=1.0e-07, max_size=20, nrest=1, out_of_core=False):
    """
    Diagonalize the similarity-transformed CCSDT Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
    from miniccpy.energy import calc_rel_dea

    remove_file("eomcc-vectors.hdf5")
    if out_of_core:
        f = h5py.File("eomcc-vectors.hdf5", "w")

    eps = np.diagonal(H1)
    n = np.newaxis
    e_abcdkl = (eps[v, n, n, n, n, n] + eps[n, v, n, n, n, n] + eps[n, n, v, n, n, n] + eps[n, n, n, v, n, n] - eps[n, n, n, n, o, n] - eps[n, n, n, n, n, o])
    e_abck = (eps[v, n, n, n] + eps[n, v, n, n] + eps[n, n, v, n] - eps[n, n, n, o])
    e_ab = (eps[v, n] + eps[n, v])

    t1, t2, t3 = T

    nunocc, nocc = t1.shape
    n1 = nunocc**2
    n2 = nunocc**3 * nocc
    n3 = nunocc**4 * nocc**2
    ndim = n1 + n2 + n3
    
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
                     R[n1:n1+n2].reshape(nunocc, nunocc, nunocc, nocc),
                     R[n1+n2:].reshape(nunocc, nunocc, nunocc, nunocc, nocc, nocc),
                     t1, t2, t3, H1, H2, o, v)

    print("    ==> DEA-EOMCCSDT iterations <==")
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
                   residual[n1:n1+n2].reshape(nunocc, nunocc, nunocc, nocc),
                   residual[n1+n2:].reshape(nunocc, nunocc, nunocc, nunocc, nocc, nocc),
                   omega,
                   e_ab,
                   e_abck,
                   e_abcdkl)
        for p in range(curr_size):
            b = B[p, :] / np.linalg.norm(B[p, :])
            q -= np.dot(b.T, q) * b
        q *= 1.0 / np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[curr_size, :] = q
            sigma[curr_size, :] = HR(q[:n1].reshape(nunocc, nunocc),
                                     q[n1:n1+n2].reshape(nunocc, nunocc, nunocc, nocc),
                                     q[n1+n2:].reshape(nunocc, nunocc, nunocc, nunocc, nocc, nocc),
                                     t1, t2, t3, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[j, :] = restart_block[:, j]
                sigma[j, :] = HR(restart_block[:n1, j].reshape(nunocc, nunocc),
                                 restart_block[n1:n1+n2, j].reshape(nunocc, nunocc, nunocc, nocc),
                                 restart_block[n1+n2:, j].reshape(nunocc, nunocc, nunocc, nunocc, nocc, nocc),
                                 t1, t2, t3, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        print("DEA-EOMCCSDT iterations did not converge")

    # Save the final converged root in an excitation tuple
    R = (R[:n1].reshape(nunocc, nunocc), R[n1:n1+n2].reshape(nunocc, nunocc, nunocc, nocc), R[n1+n2:].reshape(nunocc, nunocc, nunocc, nunocc, nocc, nocc))
    # r0 for a root in DEA is 0 by definition
    r0 = 0.0
    # Compute relative excitation level diagnostic
    rel = calc_rel_dea(R[0], R[1])
    # remove the HDF5 file
    remove_file("eomcc-vectors.hdf5")
    return R, omega, r0, rel

def update(r1, r2, r3, omega, e_ab, e_abck, e_abcdkl):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""

    for a in range(r1.shape[0]):
        for b in range(r1.shape[1]):
            denom = omega - e_ab[a, b]
            if denom == 0: continue
            r1[a, b] /= denom
    #r1 /= (omega - e_ab)
    r2 /= (omega - e_abck)
    r3 /= (omega - e_abcdkl)

    return np.hstack([r1.flatten(), r2.flatten(), r3.flatten()])

def HR(r1, r2, r3, t1, t2, t3, H1, H2, o, v):
    """Compute the matrix-vector product H * R, where
    H is the CCSD similarity-transformed Hamiltonian and R is
    the DEA-EOMCC linear excitation operator."""

    # update R1
    HR1 = build_HR1(r1, r2, r3, H1, H2, o, v)
    # update R2
    HR2 = build_HR2(r1, r2, r3, t1, t2, t3, H1, H2, o, v)
    # update R3
    HR3 = build_HR3(r1, r2, r3, t1, t2, t3, H1, H2, o, v)

    return np.hstack( [HR1.flatten(), HR2.flatten(), HR3.flatten()] )

def build_HR1(r1, r2, r3, H1, H2, o, v):
    """Compute the projection of HR on 2p excitations
        X[a, b] = < ab | [ HBar(CCSDT) * (R1 + R2 + R3) ]_C | 0 >
    """
    X1 = np.einsum("ae,eb->ab", H1[v, v], r1, optimize=True)
    X1 += 0.25 * np.einsum("abef,ef->ab", H2[v, v, v, v], r1, optimize=True)
    X1 += 0.5 * np.einsum("me,abem->ab", H1[o, v], r2, optimize=True)
    X1 += 0.5 * np.einsum("anef,ebfn->ab", H2[v, o, v, v], r2, optimize=True)
    X1 += 0.125 * np.einsum("mnef,abefmn->ab", H2[o, o, v, v], r3, optimize=True)
    # antisymmetrize A(ab)
    X1 -= np.transpose(X1, (1, 0))
    return X1

def build_HR2(r1, r2, r3, t1, t2, t3, H1, H2, o, v):
    """Compute the projection of HR on 3p-1h excitations
        X[a, b, c, k] = < kabc | [ HBar(CCSDT) * (R1 + R2 + R3) ]_C | 0 >
    """
    I_vo = (
            0.5 * np.einsum("amef,ef->am", H2[v, o, v, v], r1, optimize=True)
            + 0.5 * np.einsum("mnef,bfem->bn", H2[o, o, v, v], r2, optimize=True)
    )
    I_oo = (
            0.5 * np.einsum("mnef,ef->mn", H2[o, o, v, v], r1, optimize=True)
    )
    X2 = -(3.0 / 6.0) * np.einsum("am,bcmk->abck", I_vo, t2, optimize=True)
    X2 += (3.0 / 6.0) * np.einsum("cbke,ae->abck", H2[v, v, o, v], r1, optimize=True)
    X2 -= (1.0 / 6.0) * np.einsum("mk,abcm->abck", H1[o, o], r2, optimize=True)
    X2 += (3.0 / 6.0) * np.einsum("be,aeck->abck", H1[v, v], r2, optimize=True)
    X2 += (3.0 / 12.0) * np.einsum("abef,efck->abck", H2[v, v, v, v], r2, optimize=True)
    X2 += (3.0 / 6.0) * np.einsum("cmke,abem->abck", H2[v, o, o, v], r2, optimize=True)
    # terms contracted with T3
    X2 += (1.0 / 12.0) * np.einsum("mn,abcmnk->abck", I_oo, t3, optimize=True)
    # parts contracted with R(4h-2p)
    X2 += (1.0 / 6.0) * np.einsum("me,abcekm->abck", H1[o, v], r3, optimize=True)
    X2 += (3.0 / 12.0) * np.einsum("cnef,abefkn->abck", H2[v, o, v, v], r3, optimize=True)
    X2 -= (1.0 / 12.0) * np.einsum("mnkf,abcfmn->abck", H2[o, o, o, v], r3, optimize=True)
    # antisymmetrize A(abc)
    X2 -= np.transpose(X2, (0, 2, 1, 3)) # A(bc)
    X2 -= np.transpose(X2, (1, 0, 2, 3)) + np.transpose(X2, (2, 1, 0, 3)) # A(a/bc)
    return X2

def build_HR3(r1, r2, r3, t1, t2, t3, H1, H2, o, v):
    """Compute the projection of HR on 4p-2h excitations
        X[a, b, c, d, k, l] = < klabcd | [ HBar(CCSDT) * (R1 + R2 + R3) ]_C | 0 >
    """
    # I(am)
    I_vo = (
            0.5 * np.einsum("amef,ef->am", H2[v, o, v, v], r1, optimize=True)
            + 0.5 * np.einsum("mnef,bfem->bn", H2[o, o, v, v], r2, optimize=True)
            + np.einsum("me,ef->fm", H1[o, v], r1, optimize=True)
    )
    # I(mn)
    I_oo = (
            # 1/2 h(mnef) r1(ef)
            0.5 * np.einsum("mnef,ef->mn", H2[o, o, v, v], r1, optimize=True)
    )
    # I(abce)
    I_vvvv = (
          (3.0 / 6.0) * np.einsum("cmfe,abem->abcf", H2[v, o, v, v], r2, optimize=True)
        + (3.0 / 6.0) * np.einsum("acef,eb->abcf", H2[v, v, v, v], r1, optimize=True)
        - (1.0 / 12.0) * np.einsum("mnef,abcfmn->abce", H2[o, o, v, v], r3, optimize=True)
    )
    # antisymmetrize A(abc)
    I_vvvv -= np.transpose(I_vvvv, (0, 2, 1, 3)) # A(bc)
    I_vvvv -= np.transpose(I_vvvv, (1, 0, 2, 3)) + np.transpose(I_vvvv, (2, 1, 0, 3)) # A(a/bc)
    # I(abmk)
    I_vvoo = (
        (1.0 / 2.0) * np.einsum("nmke,abem->abnk", H2[o, o, o, v], r2, optimize=True)
        - np.einsum("bmje,ec->bcmj", H2[v, o, o, v], r1, optimize=True)
        + 0.5 * np.einsum("amfe,fbek->abmk", H2[v, o, v, v], r2, optimize=True)
        + (1.0 / 4.0) * np.einsum("mnef,abefkn->abmk", H2[o, o, v, v], r3, optimize=True)
        # contribution from 4-body HBar here
        - (1.0 / 4.0) * np.einsum("mn,bcnk->bcmk", I_oo, t2, optimize=True) # an extra factor of 1/2 applied here compensate. Net weight should be (6.0 / 48.0) in final contraction.
    )
    # antisymmetrize A(ab)
    I_vvoo -= np.transpose(I_vvoo, (1, 0, 2, 3))
    # additional intermediates for T3
    I_vvov = (
        np.einsum("bmfe,af->abme", H2[v, o, v, v], r1, optimize=True)
        + (1.0 / 2.0) * np.einsum("mnef,abfn->abme", H2[o, o, v, v], r2, optimize=True)
    )
    # antisymmetrize A(ab)
    I_vvov -= np.transpose(I_vvov, (1, 0, 2, 3))
    I_oovo = (
        - np.einsum("mnie,ae->mnai", H2[o, o, o, v], r1, optimize=True)  # (!) flipped sign from + to - here
        + 0.5 * np.einsum("mnef,efai->mnai", H2[o, o, v, v], r2, optimize=True)
    )
    #
    X3 = -(4.0 / 48.0) * np.einsum("dmlk,abcm->abcdkl", H2[v, o, o, o], r2, optimize=True)
    X3 += (12.0 / 48.0) * np.einsum("dcle,abek->abcdkl", H2[v, v, o, v], r2, optimize=True)
    X3 += (4.0 / 48.0) * np.einsum("abce,edkl->abcdkl", I_vvvv, t2, optimize=True)
    X3 -= (12.0 / 48.0) * np.einsum("abmk,cdml->abcdkl", I_vvoo, t2, optimize=True)
    # terms contracted with T3
    X3 -= (4.0 / 48.0) * np.einsum("am,bcdmkl->abcdkl", I_vo, t3, optimize=True)
    X3 += (6.0 / 48.0) * np.einsum("abme,ecdmkl->abcdkl", I_vvov, t3, optimize=True)
    X3 += (8.0 / 96.0) * np.einsum("mndl,abcmnk->abcdkl", I_oovo, t3, optimize=True)
    ####
    X3 -= (2.0 / 48.0) * np.einsum("ml,abcdkm->abcdkl", H1[o, o], r3, optimize=True)
    X3 += (4.0 / 48.0) * np.einsum("ae,ebcdkl->abcdkl", H1[v, v], r3, optimize=True)
    X3 += (1.0 / 96.0) * np.einsum("mnkl,abcdmn->abcdkl", H2[o, o, o, o], r3, optimize=True)
    X3 += (6.0 / 96.0) * np.einsum("abef,efcdkl->abcdkl", H2[v, v, v, v], r3, optimize=True)
    X3 += (8.0 / 48.0) * np.einsum("dmle,abcekm->abcdkl", H2[v, o, o, v], r3, optimize=True)
    # antisymmetrize A(abcd)A(kl)
    X3 -= np.transpose(X3, (0, 1, 2, 3, 5, 4)) # A(kl)
    X3 -= np.transpose(X3, (0, 2, 1, 3, 4, 5)) # A(bc)
    X3 -= np.transpose(X3, (1, 0, 2, 3, 4, 5)) + np.transpose(X3, (2, 1, 0, 3, 4, 5)) # A(a/bc)
    X3 -= np.transpose(X3, (3, 1, 2, 0, 4, 5)) + np.transpose(X3, (0, 3, 2, 1, 4, 5)) + np.transpose(X3, (0, 1, 3, 2, 4, 5)) # A(d/abc)
    return X3

