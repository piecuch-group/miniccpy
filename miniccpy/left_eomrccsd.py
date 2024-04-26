import time
import numpy as np
from miniccpy.energy import lccsd_energy as lcc_energy

def LT_intermediates(l2, t2):
    """Compute L2*T2-type one-body intermediates."""
    # Allocate a dictionary to store the two intermediates
    I = {"vv": None, "oo": None}
    I["vv"] = (
          -2.0 * np.einsum("afmn,efmn->ea", l2, t2, optimize=True)
          + np.einsum("afnm,efmn->ea", l2, t2, optimize=True)
    )
    I["oo"] = (
          2.0 * np.einsum("efin,efjn->ij", l2, t2, optimize=True)
          - np.einsum("efni,efjn->ij", l2, t2, optimize=True)
    )
    return I

def LH_singles(l1, l2, t2, H1, H2, I, o, v):
    """Compute the projection of the CCSD Hamiltonian on singles
        X[a, i] = < 0 | (1 + L1 + L2)*(H_N exp(T1+T2))_C | ia >
    """
    LH = np.einsum("ea,ei->ai", H1[v, v], l1, optimize=True)
    LH -= np.einsum("im,am->ai", H1[o, o], l1, optimize=True)
    LH += 2.0 * np.einsum("eima,em->ai", H2[v, o, o, v], l1, optimize=True)
    LH -= np.einsum("eiam,em->ai", H2[v, o, v, o], l1, optimize=True)
    LH += 2.0 * np.einsum("fena,efin->ai", H2[v, v, o, v], l2, optimize=True)
    LH -= np.einsum("fena,efni->ai", H2[v, v, o, v], l2, optimize=True)
    LH -= 2.0 * np.einsum("finm,afmn->ai", H2[v, o, o, o], l2, optimize=True)
    LH += np.einsum("finm,afnm->ai", H2[v, o, o, o], l2, optimize=True)
    LH -= 2.0 * np.einsum("ge,eiga->ai", I["vv"], H2[v, o, v, v], optimize=True)
    LH += np.einsum("ge,eiag->ai", I["vv"], H2[v, o, v, v], optimize=True)
    LH -= 2.0 * np.einsum("mn,nima->ai", I["oo"], H2[o, o, o, v], optimize=True)
    LH += np.einsum("mn,inma->ai", I["oo"], H2[o, o, o, v], optimize=True)
    return LH

def LH_doubles(l1, l2, t2, H1, H2, I, o, v):
    """Compute the projection of the CCSD Hamiltonian on doubles
        X[a, b, i, j] = < 0 | (1 + L2 + L2) * (H_N exp(T1+T2))_C | ijab >
    """
    LH = -np.einsum("ijmb,am->abij", H2[o, o, o, v], l1, optimize=True)
    LH += np.einsum("ejab,ei->abij", H2[v, o, v, v], l1, optimize=True)
    LH += 2.0 * np.einsum("ejmb,aeim->abij", H2[v, o, o, v], l2, optimize=True)
    LH -= np.einsum("ejmb,aemi->abij", H2[v, o, o, v], l2, optimize=True)
    LH += np.einsum("ea,ebij->abij", H1[v, v], l2, optimize=True)
    LH -= np.einsum("im,abmj->abij", H1[o, o], l2, optimize=True)
    LH += np.einsum("jb,ai->abij", H1[o, v], l1, optimize=True)
    LH += 0.5 * np.einsum("ijmn,abmn->abij", H2[o, o, o, o], l2, optimize=True)
    LH += 0.5 * np.einsum("efab,efij->abij", H2[v, v, v, v], l2, optimize=True)
    LH -= np.einsum("eiam,ebmj->abij", H2[v, o, v, o], l2, optimize=True)
    LH -= np.einsum("ejam,ebim->abij", H2[v, o, v, o], l2, optimize=True)
    LH += np.einsum("ea,ijeb->abij", I["vv"], H2[o, o, v, v], optimize=True)
    LH -= np.einsum("im,mjab->abij", I["oo"], H2[o, o, v, v], optimize=True)
    # apply symmetrizer (ij)(ab)
    LH += LH.transpose(1, 0, 3, 2)
    return LH

def update(l1, l2, omega, e_ai, e_abij):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""

    l1 /= (omega - e_ai)
    l2 /= (omega - e_abij)

    return np.hstack([l1.flatten(), l2.flatten()])

def LH(l1, l2, t1, t2, H1, H2, o, v):
    """Compute the matrix-vector product H * R, where
    H is the CCSD similarity-transformed Hamiltonian and R is
    the EOMCCSD linear excitation operator."""

    I = LT_intermediates(l2, t2)
    LH1 = LH_singles(l1, l2, t2, H1, H2, I, o, v)
    LH2 = LH_doubles(l1, l2, t2, H1, H2, I, o, v)

    return np.hstack( [LH1.flatten(), LH2.flatten()] )

def calc_LR(L, R, nocc, nunocc):
    # unpack L
    l1 = L[:nunocc*nocc].reshape(nunocc, nocc)
    l2 = L[nocc*nunocc:].reshape(nunocc, nunocc, nocc, nocc)
    # unpack R
    r1, r2 = R
    # spin-summed quantity
    l2_ss = 2.0 * l2 - np.transpose(l2, (0, 1, 3, 2))
    # compute LR
    LR = 2.0 * np.einsum("ai,ai->", l1, r1, optimize=True)
    LR += np.einsum("abij,abij->", l2_ss, r2, optimize=True)
    return LR

def kernel(R, T, omega, H1, H2, o, v, maxit=80, convergence=1.0e-07, max_size=20, nrest=1):
    """
    Diagonalize the similarity-transformed CCSD Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
    eps = np.diagonal(H1)
    n = np.newaxis
    e_abij = (eps[v, n, n, n] + eps[n, v, n, n] - eps[n, n, o, n] - eps[n, n, n, o])
    e_ai = (eps[v, n] - eps[n, o])

    t1, t2 = T

    nunocc, nocc = e_ai.shape
    n1 = nunocc * nocc
    n2 = nocc**2 * nunocc**2
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
    sigma[:, 0] = LH(L[:n1].reshape(nunocc, nocc),
                     L[n1:].reshape(nunocc, nunocc, nocc, nocc),
                     t1, t2, H1, H2, o, v)

    print("    ==> Left-EOMCCSD iterations <==")
    print("    The initial guess energy = ", omega)
    print("")
    print("     Iter               Energy                 |dE|                 |dL|")
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

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(niter, omega, delta_e, res_norm, minutes, seconds))
        if res_norm < convergence and abs(delta_e) < convergence:
            break

        # update residual vector
        q = update(residual[:n1].reshape(nunocc, nocc),
                   residual[n1:].reshape(nunocc, nunocc, nocc, nocc),
                   omega,
                   e_ai,
                   e_abij)
        for p in range(curr_size):
            b = B[:, p] / np.linalg.norm(B[:, p])
            q -= np.dot(b.T, q) * b
        q /= np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[:, curr_size] = q
            sigma[:, curr_size] = LH(q[:n1].reshape(nunocc, nocc),
                                     q[n1:].reshape(nunocc, nunocc, nocc, nocc),
                                     t1, t2, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[:, j] = restart_block[:, j]
                sigma[:, j] = LH(restart_block[:n1, j].reshape(nunocc, nocc),
                                 restart_block[n1:, j].reshape(nunocc, nunocc, nocc, nocc),
                                 t1, t2, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1
    else:
        print("Left-EOMCCSD iterations did not converge")

    # Normalize <L|R> = 1
    LR = calc_LR(L, R, nocc, nunocc)
    L /= LR
    # Save the final converged root in an excitation tuple
    L = (L[:n1].reshape(nunocc, nocc), L[n1:].reshape(nunocc, nunocc, nocc, nocc))

    return L, omega
