import time
import numpy as np

def kernel(R0, T, omega, fock, g, H1, H2, o, v, maxit=80, convergence=1.0e-07, max_size=20, nrest=1):
    """
    Diagonalize the similarity-transformed CCSDT Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
    from miniccpy.energy import calc_r0, calc_rel

    eps = np.diagonal(H1)
    f_eps = np.diagonal(fock)
    n = np.newaxis
    e_abcijk = (f_eps[v, n, n, n, n, n] + f_eps[n, v, n, n, n, n] + f_eps[n, n, v, n, n, n]
                - f_eps[n, n, n, o, n, n] - f_eps[n, n, n, n, o, n] - f_eps[n, n, n, n, n, o])
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

    # Allocate the B and sigma matrices
    sigma = np.zeros((ndim, max_size))
    B = np.zeros((ndim, max_size))
    restart_block = np.zeros((ndim, nrest))

    # Initial values
    B[:, 0] = R
    sigma[:, 0] = HR(R[:n1].reshape(nunocc, nocc),
                     R[n1:n1+n2].reshape(nunocc, nunocc, nocc, nocc),
                     R[n1+n2:].reshape(nunocc, nunocc, nunocc, nocc, nocc, nocc),
                     t1, t2, t3, fock, g, H1, H2, o, v)

    print("    ==> EOM-CC3 iterations <==")
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
                                     t1, t2, t3, fock, g, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[:, j] = restart_block[:, j]
                sigma[:, j] = HR(restart_block[:n1, j].reshape(nunocc, nocc),
                                 restart_block[n1:n1+n2, j].reshape(nunocc, nunocc, nocc, nocc),
                                 restart_block[n1+n2:, j].reshape(nunocc, nunocc, nunocc, nocc, nocc, nocc),
                                 t1, t2, t3, fock, g, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1
    else:
        print("EOM-CC3 iterations did not converge")

    # Calculate r0 for the root
    r0 = calc_r0(R[:n1].reshape(nunocc, nocc),
                 R[n1:n1+n2].reshape(nunocc, nunocc, nocc, nocc),
                 H1, H2, omega, o, v)
    # Compute relative excitation level diagnostic
    rel = calc_rel(r0,
                   R[:n1].reshape(nunocc, nocc),
                   R[n1:n1+n2].reshape(nunocc, nunocc, nocc, nocc))

    return R, omega, r0, rel

def update(r1, r2, r3, omega, e_ai, e_abij, e_abcijk):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""

    r1 /= (omega - e_ai)
    r2 /= (omega - e_abij)
    r3 /= (omega - e_abcijk)

    return np.hstack([r1.flatten(), r2.flatten(), r3.flatten()])


def HR(r1, r2, r3, t1, t2, t3, fock, g, H1, H2, o, v):
    """Compute the matrix-vector product H * R, where
    H is the CCSDT similarity-transformed Hamiltonian and R is
    the EOMCCSDT linear excitation operator."""

    # update R1
    HR1 = build_HR1(r1, r2, r3, H1, H2, o, v)
    # update R2
    HR2 = build_HR2(r1, r2, r3, t1, t2, t3, H1, H2, o, v)
    # update R3
    HR3 = build_HR3(r1, r2, r3, t1, t2, t3, fock, g, o, v)

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

def build_HR3(r1, r2, r3, t1, t2, t3, fock, g, o, v):
    """Compute the projection of HR on triples
        X[a, b, c, i, j, k] = < ijkabc | [ HBar(CCSDT) * (R1 + R2 + R3) ]_C | 0 >
    """

    # Intermediates
    Q1 = -np.einsum("mnfe,an->amef", g[o, o, v, v], t1, optimize=True)
    I_vovv = g[v, o, v, v] + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    I_ooov = g[o, o, o, v] + 0.5 * Q1

    Q1 = -np.einsum("bmfe,am->abef", I_vovv, t1, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    I_vvvv = g[v, v, v, v] + Q1

    Q1 = +np.einsum("nmje,ei->mnij", I_ooov, t1, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    I_oooo = g[o, o, o, o] + Q1


    # h(vvov) and h(vooo) intermediates resulting from exp(-T1) H_N exp(T1)
    Q1 = g[v, o, o, v] + 0.5 * np.einsum("amef,ei->amif", g[v, o, v, v], t1, optimize=True)
    Q1 = np.einsum("amif,fj->amij", Q1, t1, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    I_vooo = g[v, o, o, o] + Q1 - np.einsum("nmij,an->amij", I_oooo, t1, optimize=True)

    Q1 = g[o, v, o, v] - 0.5 * np.einsum("mnie,bn->mbie", g[o, o, o, v], t1, optimize=True)
    Q1 = -np.einsum("mbie,am->abie", Q1, t1, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    I_vvov = g[v, v, o, v] + Q1 + np.einsum("abfe,fi->abie", I_vvvv, t1, optimize=True)

    I_voov = g[v, o, o, v] + (
            - np.einsum("nmie,an->amie", g[o, o, o, v], t1, optimize=True)
            + np.einsum("amfe,fi->amie", g[v, o, v, v], t1, optimize=True)
            - np.einsum("mnef,ei,an->amie", g[o, o, v, v], t1, t1, optimize=True)
    )
    
    X_vooo = 0.5 * I_vooo + (
                + np.einsum("amie,ej->amij", I_voov, r1, optimize=True)
                - 0.5 * np.einsum("nmij,an->amij", I_oooo, r1, optimize=True)
    )
    X_vooo -= np.transpose(X_vooo, (0, 1, 3, 2))

    X_vvov = 0.5 * I_vvov + (
                - np.einsum("amie,bm->abie", I_voov, r1, optimize=True)
                + 0.5 * np.einsum("abfe,fi->abie", I_vvvv, r1, optimize=True)
    )
    X_vvov -= np.transpose(X_vvov, (1, 0, 2, 3))

    I_oooo += np.einsum("mnef,fi,ej->mnij", g[o, o, v, v], t1, t1, optimize=True)
    I_vvvv += np.einsum("mnef,am,bn->abef", g[o, o, v, v], t1, t1, optimize=True)
    #X2[v, v, o, v] =(
    #    np.einsum("amje,bm->baje", H2[v, o, o, v], r1, optimize=True)
    #    + np.einsum("amfe,bejm->bajf", H2[v, o, v, v], r2, optimize=True)
    #    + 0.5 * np.einsum("abfe,ej->bajf", H2[v, v, v, v], r1, optimize=True)
    #    + 0.25 * np.einsum("nmje,abmn->baje", H2[o, o, o, v], r2, optimize=True)
    #)
    #X2[v, v, o, v] -= np.transpose(X2[v, v, o, v], (1, 0, 2, 3))

    #X2[v, o, o, o] = (
    #    -np.einsum("bmie,ej->bmji", H2[v, o, o, v], r1, optimize=True)
    #    +np.einsum("nmie,bejm->bnji", H2[o, o, o, v], r2, optimize=True)
    #    - 0.5 * np.einsum("nmij,bm->bnji", H2[o, o, o, o], r1, optimize=True)
    #    + 0.25 * np.einsum("bmfe,efij->bmji", H2[v, o, v, v], r2, optimize=True)
    #)
    #X2[v, o, o, o] -= np.transpose(X2[v, o, o, o], (0, 1, 3, 2))

    # <ijkabc| [H(R1+R2)]_C | 0 >
    X3 = 0.25 * np.einsum("baje,ecik->abcijk", X_vvov, t2, optimize=True)
    X3 += 0.25 * np.einsum("baje,ecik->abcijk", I_vvov, r2, optimize=True)
    X3 -= 0.25 * np.einsum("bmji,acmk->abcijk", X_vooo, t2, optimize=True)
    X3 -= 0.25 * np.einsum("bmji,acmk->abcijk", I_vooo, r2, optimize=True)
    # < ijkabc | (HR3)_C | 0 >
    X3 -= (1.0 / 12.0) * np.einsum("mj,abcimk->abcijk", fock[o, o], r3, optimize=True)
    X3 += (1.0 / 12.0) * np.einsum("be,aecijk->abcijk", fock[v, v], r3, optimize=True)
    #X3 -= (1.0 / 12.0) * np.einsum("mj,abcimk->abcijk", H1[o, o], r3, optimize=True)
    #X3 += (1.0 / 12.0) * np.einsum("be,aecijk->abcijk", H1[v, v], r3, optimize=True)

    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    X3 -= np.transpose(X3, (0, 1, 2, 3, 5, 4))
    X3 -= np.transpose(X3, (0, 1, 2, 4, 3, 5)) + np.transpose(X3, (0, 1, 2, 5, 4, 3))
    X3 -= np.transpose(X3, (0, 2, 1, 3, 4, 5))
    X3 -= np.transpose(X3, (1, 0, 2, 3, 4, 5)) + np.transpose(X3, (2, 1, 0, 3, 4, 5))

    return X3

