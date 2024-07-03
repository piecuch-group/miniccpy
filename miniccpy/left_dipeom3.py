import time
import numpy as np
from miniccpy.utilities import get_memory_usage

def build_LH1(l1, l2, t2, H1, H2, o, v):
    """Compute the projection of the CCSD Hamiltonian on 2h excitations
        X[i, j] = < 0 | (L1 + L2)*(H_N exp(T1+T2))_C | ij >
    """
    I_vo = -0.5 * np.einsum("mngo,fgno->fm", l2, t2, optimize=True)

    X1 = -np.einsum("im,jm->ij", l1, H1[o, o], optimize=True)
    X1 += 0.25 * np.einsum("mn,ijmn->ij", l1, H2[o, o, o, o], optimize=True)
    X1 -= 0.5 * np.einsum("imfn,fjnm->ij", l2, H2[v, o, o, o], optimize=True)
    X1 -= 0.5 * np.einsum("fm,ijmf->ij", I_vo, H2[o, o, o, v], optimize=True)
    # antisymmetrize A(ij)
    X1 -= np.transpose(X1, (1, 0))
    return X1

def build_LH2(l1, l2, t2, H1, H2, o, v):
    """Compute the projection of the CCSD Hamiltonian on 3h1p excitations
        X[i, j, c, k] = < 0 | (L1 + L2)*(H_N exp(T1+T2))_C | ijck >
    """
    I_vo = -0.5 * np.einsum("mngo,fgno->fm", l2, t2, optimize=True)

    X2 = -0.5 * np.einsum("im,jkmc->ijck", l1, H2[o, o, o, v], optimize=True)
    X2 -= 0.5 * np.einsum("imck,jm->ijck", l2, H1[o, o], optimize=True)
    X2 += (1.0 / 6.0) * np.einsum("ijek,ec->ijck", l2, H1[v, v], optimize=True)
    X2 += 0.25 * np.einsum("mnck,ijmn->ijck", l2, H2[o, o, o, o], optimize=True)
    X2 += 0.5 * np.einsum("ijem,ekmc->ijck", l2, H2[v, o, o, v], optimize=True)
    X2 += 0.5 * np.einsum("ij,kc->ijck", l1, H1[o, v], optimize=True)
    X2 += 0.5 * np.einsum("fi,jkfc->ijck", I_vo, H2[o, o, v, v], optimize=True)
    # antisymmetrize A(ijk)
    X2 -= np.transpose(X2, (0, 3, 2, 1)) # A(jk)
    X2 -= np.transpose(X2, (1, 0, 2, 3)) + np.transpose(X2, (3, 1, 2, 0)) # A(i/jk)
    return X2

def update(l1, l2, omega, e_ij, e_ijck):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""
    l1 /= (omega - e_ij)
    l2 /= (omega - e_ijck)
    return np.hstack([l1.flatten(), l2.flatten()])

def LH(l1, l2, t1, t2, H1, H2, o, v):
    """Compute the matrix-vector product H * R, where
    H is the CCSD similarity-transformed Hamiltonian and R is
    the EOMCCSD linear excitation operator."""
    LH1 = build_LH1(l1, l2, t2, H1, H2, o, v)
    LH2 = build_LH2(l1, l2, t2, H1, H2, o, v)
    return np.hstack( [LH1.flatten(), LH2.flatten()] )

def kernel(R, T, omega, H1, H2, o, v, maxit=80, convergence=1.0e-07, max_size=20, nrest=1):
    """
    Diagonalize the similarity-transformed CCSD Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
    eps = np.diagonal(H1)
    n = np.newaxis
    e_ijck = (-eps[o, n, n, n] - eps[n, o, n, n] + eps[n, n, v, n] - eps[n, n, n, o])
    e_ij = (-eps[o, n] - eps[n, o])

    t1, t2 = T

    nunocc, nocc = t1.shape
    n1 = nocc**2
    n2 = nocc**3 * nunocc
    ndim = n1 + n2

    # Set the initial vector to be R
    r1, r2 = R
    Rvec = np.hstack([r1.flatten(), r2.flatten()])
    L = Rvec.copy()

    # Allocate the B and sigma matrices
    sigma = np.zeros((ndim, max_size))
    B = np.zeros((ndim, max_size))
    restart_block = np.zeros((ndim, nrest))

    # Initial values
    B[:, 0] = L
    sigma[:, 0] = LH(L[:n1].reshape(nocc, nocc),
                     L[n1:].reshape(nocc, nocc, nunocc, nocc),
                     t1, t2, H1, H2, o, v)

    print("    ==> Left-DIPEOMCC(3h-1p) iterations <==")
    print("    The initial guess energy = ", omega)
    print("")
    print("     Iter               Energy                 |dE|                 |dL|     Wall Time     Memory")
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

        if res_norm < convergence and abs(delta_e) < convergence:
            toc = time.time()
            minutes, seconds = divmod(toc - tic, 60)
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
            break

        # update residual vector
        q = update(residual[:n1].reshape(nocc, nocc),
                   residual[n1:].reshape(nocc, nocc, nunocc, nocc),
                   omega,
                   e_ij,
                   e_ijck)
        for p in range(curr_size):
            b = B[:, p] / np.linalg.norm(B[:, p])
            q -= np.dot(b.T, q) * b
        q /= np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[:, curr_size] = q
            sigma[:, curr_size] = LH(q[:n1].reshape(nocc, nocc),
                                     q[n1:].reshape(nocc, nocc, nunocc, nocc),
                                     t1, t2, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[:, j] = restart_block[:, j]
                sigma[:, j] = LH(restart_block[:n1, j].reshape(nocc, nocc),
                                 restart_block[n1:, j].reshape(nocc, nocc, nunocc, nocc),
                                 t1, t2, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        print("Left-DIPEOMCC(3h-1p) iterations did not converge")

    # Save the final converged root in an excitation tuple
    L = (L[:n1].reshape(nocc, nocc), L[n1:].reshape(nocc, nocc, nunocc, nocc))

    return L, omega
