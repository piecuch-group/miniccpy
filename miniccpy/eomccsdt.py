import time
import numpy as np

def kernel(R0, T, omega, H1, H2, o, v, maxit=80, convergence=1.0e-07, max_size=20, nrest=1):
    """
    Diagonalize the similarity-transformed CCSDT Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
    from miniccpy.energy import calc_r0, calc_rel

    eps = np.diagonal(H1)
    n = np.newaxis
    e_abcijk = (eps[v, n, n, n, n, n] + eps[n, v, n, n, n, n] + eps[n, n, v, n, n, n]
                - eps[n, n, n, o, n, n] - eps[n, n, n, n, o, n] - eps[n, n, n, n, n, o])
    e_abij = (eps[v, n, n, n] + eps[n, v, n, n] - eps[n, n, o, n] - eps[n, n, n, o])
    e_ai = (eps[v, n] - eps[n, o])

    t1, t2, t3 = T

    nunocc, nocc = e_ai.shape
    n1 = nunocc * nocc
    n2 = nocc**2 * nunocc**2
    n3 = nocc**3 * nunocc**3
    ndim = n1 + n2 + n3
    
    if len(R0) < ndim:
        R = np.zeros(ndim)
        R[:len(R0)] = R0
    else:
        R = R0.copy()

    # Allocate the B and sigma matrices
    sigma = np.zeros((ndim, max_size))
    B = np.zeros((ndim, max_size))
    restart_block = np.zeros((ndim, nrest))

    # Initial values
    B[:, 0] = R
    sigma[:, 0] = HR(R[:n1].reshape(nunocc, nocc),
                     R[n1:n1+n2].reshape(nunocc, nunocc, nocc, nocc),
                     R[n1+n2:].reshape(nunocc, nunocc, nunocc, nocc, nocc, nocc),
                     t1, t2, t3, H1, H2, o, v)

    print("    ==> EOMCCSDT iterations <==")
    print("    The initial guess energy = ", omega)
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
        q = update(residual[:n1].reshape(nunocc, nocc),
                   residual[n1:n1+n2].reshape(nunocc, nunocc, nocc, nocc),
                   residual[n1+n2:].reshape(nunocc, nunocc, nunocc, nocc, nocc, nocc),
                   omega,
                   e_ai,
                   e_abij,
                   e_abcijk)
        for p in range(curr_size):
            b = B[:, p] / np.linalg.norm(B[:, p])
            q -= np.dot(b.T, q) * b
        q *= 1.0 / np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[:, curr_size] = q
            sigma[:, curr_size] = HR(q[:n1].reshape(nunocc, nocc),
                                     q[n1:n1+n2].reshape(nunocc, nunocc, nocc, nocc),
                                     q[n1+n2:].reshape(nunocc, nunocc, nunocc, nocc, nocc, nocc),
                                     t1, t2, t3, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[:, j] = restart_block[:, j]
                sigma[:, j] = HR(restart_block[:n1, j].reshape(nunocc, nocc),
                                 restart_block[n1:n1+n2, j].reshape(nunocc, nunocc, nocc, nocc),
                                 restart_block[n1+n2:, j].reshape(nunocc, nunocc, nunocc, nocc, nocc, nocc),
                                 t1, t2, t3, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1
    else:
        print("EOMCCSDT iterations did not converge")

    # Save the final converged root in an excitation tuple
    R = (R[:n1].reshape(nunocc, nocc), R[n1:n1+n2].reshape(nunocc, nunocc, nocc, nocc), R[n1+n2:].reshape(nunocc, nunocc, nunocc, nocc, nocc, nocc))
    # Calculate r0 for the root
    r0 = calc_r0(R[0], R[1], H1, H2, omega, o, v)
    # Compute relative excitation level diagnostic
    rel = calc_rel(r0, R[0], R[1])
    return R, omega, r0, rel

def update(r1, r2, r3, omega, e_ai, e_abij, e_abcijk):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""

    r1 /= (omega - e_ai)
    r2 /= (omega - e_abij)
    r3 /= (omega - e_abcijk)

    return np.hstack([r1.flatten(), r2.flatten(), r3.flatten()])


def HR(r1, r2, r3, t1, t2, t3, H1, H2, o, v):
    """Compute the matrix-vector product H * R, where
    H is the CCSDT similarity-transformed Hamiltonian and R is
    the EOMCCSDT linear excitation operator."""

    # update R1
    HR1 = build_HR1(r1, r2, r3, H1, H2, o, v)
    # update R2
    HR2 = build_HR2(r1, r2, r3, t1, t2, t3, H1, H2, o, v)
    # update R3
    HR3 = build_HR3(r1, r2, r3, t1, t2, t3, H1, H2, o, v)

    return np.hstack( [HR1.flatten(), HR2.flatten(), HR3.flatten()] )


def build_HR1(r1, r2, r3, H1, H2, o, v):
    """Compute the projection of HR on singles
        X[a, i] = < ia | [ HBar(CCSDT) * (R1 + R2 + R3) ]_C | 0 >
    """

    X1 = -np.einsum("mi,am->ai", H1[o, o], r1, optimize=True)
    X1 += np.einsum("ae,ei->ai", H1[v, v], r1, optimize=True)
    X1 += np.einsum("amie,em->ai", H2[v, o, o, v], r1, optimize=True)
    X1 -= 0.5 * np.einsum("mnif,afmn->ai", H2[o, o, o, v], r2, optimize=True)
    X1 += 0.5 * np.einsum("anef,efin->ai", H2[v, o, v, v], r2, optimize=True)
    X1 += np.einsum("me,aeim->ai", H1[o, v], r2, optimize=True)
    X1 += 0.25 * np.einsum("mnef,aefimn->ai", H2[o, o, v, v], r3, optimize=True)
    return X1


def build_HR2(r1, r2, r3, t1, t2, t3, H1, H2, o, v):
    """Compute the projection of HR on doubles
        X[a, b, i, j] = < ijab | [ HBar(CCSDT) * (R1 + R2 + R3) ]_C | 0 >
    """

    X2 = -0.5 * np.einsum("mi,abmj->abij", H1[o, o], r2, optimize=True)  # A(ij)
    X2 += 0.5 * np.einsum("ae,ebij->abij", H1[v, v], r2, optimize=True)  # A(ab)
    X2 += 0.5 * 0.25 * np.einsum("mnij,abmn->abij", H2[o, o, o, o], r2, optimize=True)
    X2 += 0.5 * 0.25 * np.einsum("abef,efij->abij", H2[v, v, v, v], r2, optimize=True)
    X2 += np.einsum("amie,ebmj->abij", H2[v, o, o, v], r2, optimize=True)  # A(ij)A(ab)
    X2 -= 0.5 * np.einsum("bmji,am->abij", H2[v, o, o, o], r1, optimize=True)  # A(ab)
    X2 += 0.5 * np.einsum("baje,ei->abij", H2[v, v, o, v], r1, optimize=True)  # A(ij)

    Q1 = -0.5 * np.einsum("mnef,bfmn->eb", H2[o, o, v, v], r2, optimize=True)
    X2 += 0.5 * np.einsum("eb,aeij->abij", Q1, t2, optimize=True)  # A(ab)

    Q1 = 0.5 * np.einsum("mnef,efjn->mj", H2[o, o, v, v], r2, optimize=True)
    X2 -= 0.5 * np.einsum("mj,abim->abij", Q1, t2, optimize=True)  # A(ij)

    Q1 = np.einsum("amfe,em->af", H2[v, o, v, v], r1, optimize=True)
    X2 += 0.5 * np.einsum("af,fbij->abij", Q1, t2, optimize=True)  # A(ab)
    Q2 = np.einsum("nmie,em->ni", H2[o, o, o, v], r1, optimize=True)
    X2 -= 0.5 * np.einsum("ni,abnj->abij", Q2, t2, optimize=True)  # A(ij)

    I_ov = np.einsum("mnef,fn->me", H2[o, o, v, v], r1, optimize=True)
    X2 += 0.25 * np.einsum("me,abeijm->abij", I_ov, t3, optimize=True)

    X2 += 0.25 * np.einsum("me,abeijm->abij", H1[o, v], r3, optimize=True)
    X2 -= 0.5 * 0.5 * np.einsum("mnjf,abfimn->abij", H2[o, o, o, v], r3, optimize=True)
    X2 += 0.5 * 0.5 * np.einsum("bnef,aefijn->abij", H2[v, o, v, v], r3, optimize=True)

    X2 -= np.transpose(X2, (0, 1, 3, 2))
    X2 -= np.transpose(X2, (1, 0, 2, 3))

    return X2

def build_HR3(r1, r2, r3, t1, t2, t3, H1, H2, o, v):
    """Compute the projection of HR on triples
        X[a, b, c, i, j, k] = < ijkabc | [ HBar(CCSDT) * (R1 + R2 + R3) ]_C | 0 >
    """

    # Intermediates
    X1 = np.zeros_like(H1)
    X2 = np.zeros_like(H2)

    X1[o, v] = np.einsum("mnef,fn->me", H2[o, o, v, v], r1, optimize=True)

    X1[o, o] = (
            np.einsum("me,ej->mj", H1[o, v], r1, optimize=True)
            + np.einsum("mnjf,fn->mj", H2[o, o, o, v], r1, optimize=True)
            + 0.5 * np.einsum("mnef,efjn->mj", H2[o, o, v, v], r2, optimize=True)
    )

    X1[v, v] = (
            -1.0 * np.einsum("me,bm->be", H1[o, v], r1, optimize=True)
            + np.einsum("bnef,fn->be", H2[v, o, v, v], r1, optimize=True)
            - 0.5 * np.einsum("mnef,bfmn->be", H2[o, o, v, v], r2, optimize=True)
    )
    X2[o, o, o, o] = (
        np.einsum("nmje,ei->mnij", H2[o, o, o, v], r1, optimize=True)
        + 0.25 * np.einsum("mnef,efij->mnij", H2[o, o, v, v], r2, optimize=True)
    )
    X2[o, o, o, o] -= np.transpose(X2[o, o, o, o], (0, 1, 3, 2))
    X2[v, v, v, v] = (
        -1.0 * np.einsum("amef,bm->abef", H2[v, o, v, v], r1, optimize=True)
        + 0.25 * np.einsum("mnef,abmn->abef", H2[o, o, v, v], r2, optimize=True)
    )
    X2[v, v, v, v] -= np.transpose(X2[v, v, v, v], (1, 0, 2, 3))
    X2[v, o, o, v] = (
        -1.0 * np.einsum("nmje,bn->bmje", H2[o, o, o, v], r1, optimize=True)
        + np.einsum("bmfe,fj->bmje", H2[v, o, v, v], r1, optimize=True)
        + np.einsum("mnef,fcnk->cmke", H2[o, o, v, v], r2, optimize=True)
    )
    X2[v, v, o, v] =(
        np.einsum("amje,bm->baje", H2[v, o, o, v], r1, optimize=True)
        + np.einsum("amfe,bejm->bajf", H2[v, o, v, v], r2, optimize=True)
        + 0.5 * np.einsum("abfe,ej->bajf", H2[v, v, v, v], r1, optimize=True)
        + 0.25 * np.einsum("nmje,abmn->baje", H2[o, o, o, v], r2, optimize=True)
        - 0.5 * np.einsum("me,abmj->baje", X1[o, v], t2, optimize=True) 
    )
    X2[v, v, o, v] -= np.transpose(X2[v, v, o, v], (1, 0, 2, 3))
    X2[v, v, o, v] -= 0.5 * np.einsum("mnef,abfimn->abie", H2[o, o, v, v], r3, optimize=True)

    X2[v, o, o, o] = (
        -np.einsum("bmie,ej->bmji", H2[v, o, o, v], r1, optimize=True)
        +np.einsum("nmie,bejm->bnji", H2[o, o, o, v], r2, optimize=True)
        - 0.5 * np.einsum("nmij,bm->bnji", H2[o, o, o, o], r1, optimize=True)
        + 0.25 * np.einsum("bmfe,efij->bmji", H2[v, o, v, v], r2, optimize=True)
    )
    X2[v, o, o, o] -= np.transpose(X2[v, o, o, o], (0, 1, 3, 2))
    X2[v, o, o, o] += 0.5 * np.einsum("mnef,efcjnk->cmkj", H2[o, o, v, v], r3, optimize=True)

    # <ijkabc| [H(R1+R2)]_C | 0 >
    X3 = 0.25 * np.einsum("baje,ecik->abcijk", X2[v, v, o, v], t2, optimize=True)
    X3 += 0.25 * np.einsum("baje,ecik->abcijk", H2[v, v, o, v], r2, optimize=True)
    X3 -= 0.25 * np.einsum("bmji,acmk->abcijk", X2[v, o, o, o], t2, optimize=True)
    X3 -= 0.25 * np.einsum("bmji,acmk->abcijk", H2[v, o, o, o], r2, optimize=True)
    # additional terms with T3 in <ijkabc|[ H(R1+R2)]_C | 0>
    X3 += (1.0 / 12.0) * np.einsum("be,aecijk->abcijk", X1[v, v], t3, optimize=True)
    X3 -= (1.0 / 12.0) * np.einsum("mj,abcimk->abcijk", X1[o, o], t3, optimize=True)
    X3 += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", X2[o, o, o, o], t3, optimize=True)
    X3 += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", X2[v, v, v, v], t3, optimize=True)
    X3 += 0.25 * np.einsum("bmje,aecimk->abcijk", X2[v, o, o, v], t3, optimize=True)
    # < ijkabc | (HR3)_C | 0 >
    X3 -= (1.0 / 12.0) * np.einsum("mj,abcimk->abcijk", H1[o, o], r3, optimize=True)
    X3 += (1.0 / 12.0) * np.einsum("be,aecijk->abcijk", H1[v, v], r3, optimize=True)
    X3 += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H2[o, o, o, o], r3, optimize=True)
    X3 += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H2[v, v, v, v], r3, optimize=True)
    X3 += 0.25 * np.einsum("amie,ebcmjk->abcijk", H2[v, o, o, v], r3, optimize=True)

    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    X3 -= np.transpose(X3, (0, 1, 2, 3, 5, 4))
    X3 -= np.transpose(X3, (0, 1, 2, 4, 3, 5)) + np.transpose(X3, (0, 1, 2, 5, 4, 3))
    X3 -= np.transpose(X3, (0, 2, 1, 3, 4, 5))
    X3 -= np.transpose(X3, (1, 0, 2, 3, 4, 5)) + np.transpose(X3, (2, 1, 0, 3, 4, 5))

    return X3

