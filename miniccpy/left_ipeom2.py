import time
import numpy as np
from miniccpy.utilities import get_memory_usage

def build_LH1(l1, l2, H1, H2, o, v):
    """Compute the projection of the CCSD Hamiltonian on 1h excitations
        X[i] = < 0 | (L1 + L2)*(H_N exp(T1+T2))_C | i >
    """
    LH = -1.0 * np.einsum("m,im->i", l1, H1[o, o], optimize=True)
    LH -= 0.5 * np.einsum("mfn,finm->i", l2, H2[v, o, o, o], optimize=True)
    return LH


def build_LH2(l1, l2, t2, H1, H2, o, v):
    """Compute the projection of the CCSD Hamiltonian on 2h1p excitations
        X[i, b, j] = < 0 | (L1 + L2)*(H_N exp(T1+T2))_C | ibj >
    """
    LH = np.einsum("i,jb->ibj", l1, H1[o, v], optimize=True)
    LH -= 0.5 * np.einsum("m,ijmb->ibj", l1, H2[o, o, o, v], optimize=True)
    LH += 0.5 * np.einsum("iej,eb->ibj", l2, H1[v, v], optimize=True)
    LH -= np.einsum("ibm,jm->ibj", l2, H1[o, o], optimize=True)
    LH += 0.25 * np.einsum("mbn,ijmn->ibj", l2, H2[o, o, o, o], optimize=True)
    LH += np.einsum("iem,ejmb->ibj", l2, H2[v, o, o, v], optimize=True)
    I1 = -0.5 * np.einsum("mfn,efmn->e", l2, t2, optimize=True)
    LH += 0.5 * np.einsum("e,ijeb->ibj", I1, H2[o, o, v, v], optimize=True)
    LH -= np.transpose(LH, (2, 1, 0))
    return LH

def update(l1, l2, omega, e_i, e_ibj):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""
    for i, d_i in enumerate(e_i):
        denom = omega - d_i
        if denom == 0: continue
        l1[i] /= denom
    l2 /= (omega - e_ibj)
    return np.hstack([l1.flatten(), l2.flatten()])

def LH(l1, l2, t1, t2, H1, H2, o, v):
    """Compute the matrix-vector product H * R, where
    H is the CCSD similarity-transformed Hamiltonian and R is
    the EOMCCSD linear excitation operator."""

    LH1 = build_LH1(l1, l2, H1, H2, o, v)
    LH2 = build_LH2(l1, l2, t2, H1, H2, o, v)

    return np.hstack( [LH1.flatten(), LH2.flatten()] )

def calc_LR(L, R, nocc, nunocc):
    # unpack L
    l1 = L[:nocc]
    l2 = L[nocc:].reshape(nocc, nunocc, nocc)
    # unpack R
    r1, r2 = R
    # compute LR
    LR = np.einsum("i,i->", l1, r1, optimize=True)
    LR += (1.0 / 2.0) * np.einsum("ibj,ibj->", l2, r2, optimize=True)
    return LR

def kernel(R, T, omega, H1, H2, o, v, maxit=80, convergence=1.0e-07, max_size=20, nrest=1):
    """
    Diagonalize the similarity-transformed CCSD Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
    eps = np.diagonal(H1)
    n = np.newaxis
    e_ibj = (-eps[o, n, n] + eps[n, v, n] - eps[n, n, o])
    e_i = eps[o]

    t1, t2 = T

    nunocc, nocc = t1.shape
    n1 = nocc
    n2 = nocc**2 * nunocc
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
    sigma[:, 0] = LH(L[:n1].reshape(nocc),
                     L[n1:].reshape(nocc, nunocc, nocc),
                     t1, t2, H1, H2, o, v)

    print("    ==> Left-IPEOMCC(2h-1p) iterations <==")
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
        q = update(residual[:n1].reshape(nocc),
                   residual[n1:].reshape(nocc, nunocc, nocc),
                   omega,
                   e_i,
                   e_ibj)
        for p in range(curr_size):
            b = B[:, p] / np.linalg.norm(B[:, p])
            q -= np.dot(b.T, q) * b
        q /= np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[:, curr_size] = q
            sigma[:, curr_size] = LH(q[:n1].reshape(nocc),
                                     q[n1:].reshape(nocc, nunocc, nocc),
                                     t1, t2, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[:, j] = restart_block[:, j]
                sigma[:, j] = LH(restart_block[:n1, j].reshape(nocc),
                                 restart_block[n1:, j].reshape(nocc, nunocc, nocc),
                                 t1, t2, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        print("Left-IPEOMCC(2h-1p) iterations did not converge")

    # Normalize <L|R> = 1
    LR = calc_LR(L, R, nocc, nunocc)
    L /= LR
    # Save the final converged root in an excitation tuple
    L = (L[:n1].reshape(nocc), L[n1:].reshape(nocc, nunocc, nocc))

    return L, omega
