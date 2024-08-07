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
    from miniccpy.energy import calc_rel_ip

    remove_file("eomcc-vectors.hdf5")
    if out_of_core:
        f = h5py.File("eomcc-vectors.hdf5", "w")

    eps = np.diagonal(H1)
    n = np.newaxis
    e_ibcjk = (-eps[o, n, n, n, n] + eps[n, v, n, n, n] + eps[n, n, v, n, n] - eps[n, n, n, o, n] - eps[n, n, n, n, o])
    e_ibj = (-eps[o, n, n] + eps[n, v, n] - eps[n, n, o])
    e_i = -eps[o]

    t1, t2, t3 = T

    nunocc, nocc = t1.shape
    n1 = nocc
    n2 = nocc**2 * nunocc
    n3 = nocc**3 * nunocc**2
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
    sigma[0, :] = HR(R[:n1].reshape(nocc),
                     R[n1:n1+n2].reshape(nocc, nunocc, nocc),
                     R[n1+n2:].reshape(nocc, nunocc, nunocc, nocc, nocc),
                     t1, t2, t3, H1, H2, o, v)

    print("    ==> IP-EOMCCSDT iterations <==")
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
        q = update(residual[:n1].reshape(nocc),
                   residual[n1:n1+n2].reshape(nocc, nunocc, nocc),
                   residual[n1+n2:].reshape(nocc, nunocc, nunocc, nocc, nocc),
                   omega,
                   e_i,
                   e_ibj,
                   e_ibcjk)
        for p in range(curr_size):
            b = B[p, :] / np.linalg.norm(B[p, :])
            q -= np.dot(b.T, q) * b
        q *= 1.0 / np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[curr_size, :] = q
            sigma[curr_size, :] = HR(q[:n1].reshape(nocc),
                                     q[n1:n1+n2].reshape(nocc, nunocc, nocc),
                                     q[n1+n2:].reshape(nocc, nunocc, nunocc, nocc, nocc),
                                     t1, t2, t3, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[j, :] = restart_block[:, j]
                sigma[j, :] = HR(restart_block[:n1, j].reshape(nocc),
                                 restart_block[n1:n1+n2, j].reshape(nocc, nunocc, nocc),
                                 restart_block[n1+n2:, j].reshape(nocc, nunocc, nunocc, nocc, nocc),
                                 t1, t2, t3, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        print("IP-EOMCCSDT iterations did not converge")

    # Save the final converged root in an excitation tuple
    R = (R[:n1].reshape(nocc), R[n1:n1+n2].reshape(nocc, nunocc, nocc), R[n1+n2:].reshape(nocc, nunocc, nunocc, nocc, nocc))
    # Set the r0 trivially to 0
    r0 = 0.0
    # Compute the REL metric
    rel = calc_rel_ip(R[0], R[1])
    # remove the HDF5 file
    remove_file("eomcc-vectors.hdf5")
    return R, omega, r0, rel

def update(r1, r2, r3, omega, e_i, e_ibj, e_ibcjk):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""

    for i, d_i in enumerate(e_i):
        denom = omega - d_i
        if denom == 0: continue
        r1[i] /= denom
    r2 /= (omega - e_ibj)
    r3 /= (omega - e_ibcjk)

    return np.hstack([r1.flatten(), r2.flatten(), r3.flatten()])


def HR(r1, r2, r3, t1, t2, t3, H1, H2, o, v):
    """Compute the matrix-vector product H * R, where
    H is the CCSD similarity-transformed Hamiltonian and R is
    the IP-EOMCC linear excitation operator."""

    # update R1
    HR1 = build_HR1(r1, r2, r3, H1, H2, o, v)
    # update R2
    HR2 = build_HR2(r1, r2, r3, t1, t2, t3, H1, H2, o, v)
    # update R2
    HR3 = build_HR3(r1, r2, r3, t1, t2, t3, H1, H2, o, v)

    return np.hstack( [HR1.flatten(), HR2.flatten(), HR3.flatten()] )


def build_HR1(r1, r2, r3, H1, H2, o, v):
    """Compute the projection of HR on 1h excitations
        X[i] = < i | [ HBar(CCSD) * (R1 + R2 + R3) ]_C | 0 >
    """
    X1 = -np.einsum("mi,m->i", H1[o, o], r1, optimize=True)
    X1 -= 0.5 * np.einsum("mnif,mfn->i", H2[o, o, o, v], r2, optimize=True)
    X1 += np.einsum("me,iem->i", H1[o, v], r2, optimize=True)
    X1 += 0.25 * np.einsum("mnef,iefmn->i", H2[o, o, v, v], r3, optimize=True)
    return X1


def build_HR2(r1, r2, r3, t1, t2, t3, H1, H2, o, v):
    """Compute the projection of HR on 2h-1p excitations
        X[i, b, j] = < ijb | [ HBar(CCSD) * (R1 + R2 + R3) ]_C | 0 >
    """
    I1_v = -0.5 * np.einsum("mnef,mfn->e", H2[o, o, v, v], r2, optimize=True)
    # additional intermediates for T3
    I2_vvo = -np.einsum("mnef,m->efn", H2[o, o, v, v], r1, optimize=True)

    X2 = -0.5 * np.einsum("bmji,m->ibj", H2[v, o, o, o], r1, optimize=True)
    X2 += 0.5 * np.einsum("be,iej->ibj", H1[v, v], r2, optimize=True)
    X2 += 0.25 * np.einsum("mnij,mbn->ibj", H2[o, o, o, o], r2, optimize=True)
    X2 += 0.5 * np.einsum("e,ebij->ibj", I1_v, t2, optimize=True)
    X2 -= np.einsum("mi,mbj->ibj", H1[o, o], r2, optimize=True)
    X2 += np.einsum("bmje,iem->ibj", H2[v, o, o, v], r2, optimize=True)
    X2 += 0.5 * np.einsum("me,ibejm->ibj", H1[o, v], r3, optimize=True)
    X2 += 0.25 * np.einsum("bnef,iefjn->ibj", H2[v, o, v, v], r3, optimize=True)
    X2 -= 0.5 * np.einsum("mnjf,ibfmn->ibj", H2[o, o, o, v], r3, optimize=True)
    # parts connected with T3 -> Don't include; these are part of h(vvov)*R1 and h(vooo)*R1,
    # which are taken care of in CCSDT HBar T3 terms
    # X2 += 0.25 * np.einsum("efn,ebfijn->ibj", I2_vvo, t3, optimize=True)
    # antisymmetrize
    X2 -= np.transpose(X2, (2, 1, 0))
    return X2

def build_HR3(r1, r2, r3, t1, t2, t3, H1, H2, o, v):
    """Compute the projection of HR on 3h-2p excitations
        X[i, b, c, j, k] = < ijkbc | [ HBar(CCSD) * (R1 + R2 + R3) ]_C | 0 >
    """
    # Intermediates
    I2_ooo = (
            0.25 * np.einsum("mnef,iefjn->mij", H2[o, o, v, v], r3, optimize=True)
            + np.einsum("mnjf,ifn->mij", H2[o, o, o, v], r2, optimize=True)
            - 0.5 * np.einsum("mnji,n->mij", H2[o, o, o, o], r1, optimize=True)
    )
    I2_ooo -= np.transpose(I2_ooo, (0, 2, 1))
    I2_ovv = (
            -0.5 * np.einsum("mnef,ibfmn->ibe", H2[o, o, v, v], r3, optimize=True)
            + np.einsum("bnef,ifn->ibe", H2[v, o, v, v], r2, optimize=True)
            + 0.5 * np.einsum("nmie,nbm->ibe", H2[o, o, o, v], r2, optimize=True)
            + np.einsum("bmie,m->ibe", H2[v, o, o, v], r1, optimize=True)
    )
    # Additional intermediates for T3
    I1_v = -0.5 * np.einsum("mnef,mfn->e", H2[o, o, v, v], r2, optimize=True)
    I2_oov = (
        -np.einsum("nmie,n->ime", H2[o, o, o, v], r1, optimize=True)
        +np.einsum("mnef,ifn->ime", H2[o, o, v, v], r2, optimize=True)
    )
    I2_vvv = (
        -np.einsum("anef,n->aef", H2[v, o, v, v], r1, optimize=True)
        +0.5 * np.einsum("mnef,nam->aef", H2[o, o, v, v], r2, optimize=True)
    )

    X3 = -(3.0 / 12.0) * np.einsum("mj,ibcmk->ibcjk", H1[o, o], r3, optimize=True)
    X3 += (2.0 / 12.0) * np.einsum("be,iecjk->ibcjk", H1[v, v], r3, optimize=True)
    X3 += (3.0 / 24.0) * np.einsum("mnjk,ibcmn->ibcjk", H2[o, o, o, o], r3, optimize=True)
    X3 += (1.0 / 24.0) * np.einsum("bcef,iefjk->ibcjk", H2[v, v, v, v], r3, optimize=True)
    X3 += (6.0 / 12.0) * np.einsum("bmje,iecmk->ibcjk", H2[v, o, o, v], r3, optimize=True)
    X3 -= (6.0 / 12.0) * np.einsum("cmkj,ibm->ibcjk", H2[v, o, o, o], r2, optimize=True)
    X3 += (3.0 / 12.0) * np.einsum("cbke,iej->ibcjk", H2[v, v, o, v], r2, optimize=True)
    X3 -= (3.0 / 12.0) * np.einsum("mij,bcmk->ibcjk", I2_ooo, t2, optimize=True)
    X3 += (6.0 / 12.0) * np.einsum("ibe,ecjk->ibcjk", I2_ovv, t2, optimize=True)
    # parts connected with T3
    X3 += (1.0 / 12.0) * np.einsum("e,ebcijk->ibcjk", I1_v, t3, optimize=True)
    X3 += (3.0 / 12.0) * np.einsum("ime,ebcmjk->ibcjk", I2_oov, t3, optimize=True)
    X3 += (2.0 / 24.0) * np.einsum("bef,fecijk->ibcjk", I2_vvv, t3, optimize=True)
    # antisymmetrize
    X3 -= np.transpose(X3, (3, 1, 2, 0, 4)) + np.transpose(X3, (4, 1, 2, 3, 0)) # antisymmetrize A(i/jk)
    X3 -= np.transpose(X3, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    X3 -= np.transpose(X3, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    return X3
