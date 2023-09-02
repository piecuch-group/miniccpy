import time
import numpy as np

def kernel(R0, T, omega, H1, H2, o, v, maxit=80, convergence=1.0e-07):
    """
    Diagonalize the similarity-transformed CCSD Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
    from miniccpy.energy import calc_rel_ea

    eps = np.diagonal(H1)
    n = np.newaxis
    e_abcjk = (eps[v, n, n, n, n] + eps[n, v, n, n, n] + eps[n, n, v, n, n] - eps[n, n, n, o, n] - eps[n, n, n, n, o])
    e_abj = (eps[v, n, n] + eps[n, v, n] - eps[n, n, o])
    e_a = eps[v]

    t1, t2 = T

    nunocc, nocc = t1.shape
    n1 = nunocc
    n2 = nocc * nunocc**2
    n3 = nocc**2 * nunocc**3
    ndim = n1 + n2 + n3
    
    if len(R0) < ndim:
        R = np.zeros(ndim)
        R[:len(R0)] = R0

    # Allocate the B and sigma matrices
    sigma = np.zeros((ndim, maxit))
    B = np.zeros((ndim, maxit))

    # Initial values
    B[:, 0] = R
    sigma[:, 0] = HR(R[:n1].reshape(nunocc),
                     R[n1:n1+n2].reshape(nunocc, nunocc, nocc),
                     R[n1+n2:].reshape(nunocc, nunocc, nunocc, nocc, nocc),
                     t1, t2, H1, H2, o, v)

    print("    ==> EA-EOMCC(3p-2h) iterations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dR|")
    for curr_size in range(1, maxit+1):
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

        # calculate residual vector
        residual = np.dot(sigma[:, :curr_size], alpha) - omega * R
        res_norm = np.linalg.norm(residual)
        delta_e = omega - omega_old

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(curr_size, omega, delta_e, res_norm, minutes, seconds))
        if res_norm < convergence and abs(delta_e) < convergence:
            break

        # update residual vector
        q = update(residual[:n1].reshape(nunocc),
                   residual[n1:n1+n2].reshape(nunocc, nunocc, nocc),
                   residual[n1+n2:].reshape(nunocc, nunocc, nunocc, nocc, nocc),
                   omega,
                   e_a,
                   e_abj,
                   e_abcjk)
        for p in range(curr_size):
            b = B[:, p] / np.linalg.norm(B[:, p])
            q -= np.dot(b.T, q) * b
        q *= 1.0 / np.linalg.norm(q)

        B[:, curr_size] = q
        sigma[:, curr_size] = HR(q[:n1].reshape(nunocc),
                                 q[n1:n1+n2].reshape(nunocc, nunocc, nocc),
                                 q[n1+n2:].reshape(nunocc, nunocc, nunocc, nocc, nocc),
                                 t1, t2, H1, H2, o, v)
    else:
        raise ValueError("EA-EOMCC(3p-2h) iterations did not converge")

    # Set the r0 and rel trivially to 0
    r0 = 0.0
    rel = calc_rel_ea(R[:n1].reshape(nunocc),
                      R[n1:n1+n2].reshape(nunocc, nunocc, nocc))
    return R, omega, r0, rel

def update(r1, r2, r3, omega, e_a, e_abj, e_abcjk):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""

    #for a, d_a in enumerate(e_a):
    #    denom = omega - d_a
    #    if denom == 0: continue
    #    r1[a] /= denom
    r1 /= (omega - e_a)
    r2 /= (omega - e_abj)
    r3 /= (omega - e_abcjk)

    return np.hstack([r1.flatten(), r2.flatten(), r3.flatten()])


def HR(r1, r2, r3, t1, t2, H1, H2, o, v):
    """Compute the matrix-vector product H * R, where
    H is the CCSD similarity-transformed Hamiltonian and R is
    the EA-EOMCC linear excitation operator."""

    # update R1
    HR1 = build_HR1(r1, r2, r3, H1, H2, o, v)
    # update R2
    HR2 = build_HR2(r1, r2, r3, t1, t2, H1, H2, o, v)
    # update R3
    HR3 = build_HR3(r1, r2, r3, t1, t2, H1, H2, o, v)

    return np.hstack( [HR1.flatten(), HR2.flatten(), HR3.flatten()] )


def build_HR1(r1, r2, r3, H1, H2, o, v):
    """Compute the projection of HR on 1p excitations
        X[a] = < a | [ HBar(CCSD) * (R1 + R2 + R3) ]_C | 0 >
    """
    X1 = np.einsum("ae,e->a", H1[v, v], r1, optimize=True)
    X1 += 0.5 * np.einsum("anef,efn->a", H2[v, o, v, v], r2, optimize=True)
    X1 += np.einsum("me,aem->a", H1[o, v], r2, optimize=True)
    X1 += 0.25 * np.einsum("mnef,aefmn->a", H2[o, o, v, v], r3, optimize=True)
    return X1


def build_HR2(r1, r2, r3, t1, t2, H1, H2, o, v):
    """Compute the projection of HR on 2p-1h excitations
        X[a, b, j] = < jab | [ HBar(CCSD) * (R1 + R2 + R3) ]_C | 0 >
    """
    X2 = 0.5 * np.einsum("baje,e->abj", H2[v, v, o, v], r1, optimize=True)
    X2 -= 0.5 * np.einsum("mj,abm->abj", H1[o, o], r2, optimize=True)
    X2 += 0.25 * np.einsum("abef,efj->abj", H2[v, v, v, v], r2, optimize=True)
    I1 = 0.5 * np.einsum("mnef,efn->m", H2[o, o, v, v], r2, optimize=True)
    X2 -= 0.5 * np.einsum("m,abmj->abj", I1, t2, optimize=True)
    X2 += np.einsum("ae,ebj->abj", H1[v, v], r2, optimize=True)
    X2 += np.einsum("bmje,aem->abj", H2[v, o, o, v], r2, optimize=True)
    X2 += 0.5 * np.einsum("me,abejm->abj", H1[o, v], r3, optimize=True)
    X2 += 0.5 * np.einsum("bmef,aefjm->abj", H2[v, o, v, v], r3, optimize=True)
    X2 -= 0.25 * np.einsum("mnje,abemn->abj", H2[o, o, o, v], r3, optimize=True)
    X2 -= np.transpose(X2, (1, 0, 2))
    return X2

def build_HR3(r1, r2, r3, t1, t2, H1, H2, o, v):
    """Compute the projection of HR on 3p-2h excitations
        X[a, b, c, j, k] = < jkabc | [ HBar(CCSD) * (R1 + R2 + R3) ]_C | 0 >
    """
    I2_voo = (
                -np.einsum("amje,e->amj", H2[v, o, o, v], r1, optimize=True) # (!)
                + 0.5 * np.einsum("amef,efj->amj", H2[v, o, v, v], r2, optimize=True)
                + np.einsum("mnje,aen->amj", H2[o, o, o, v], r2, optimize=True)
                + 0.5 * np.einsum("mnef,aefjn->amj", H2[o, o, v, v], r3, optimize=True)
    )
    I2_vvv = (
                0.5 * np.einsum("bcef,e->bcf", H2[v, v, v, v], r1, optimize=True)
                - np.einsum("cmef,bem->bcf", H2[v, o, v, v], r2, optimize=True)
                - 0.25 * np.einsum("mnef,becmn->bcf", H2[o, o, v, v], r3, optimize=True)
    )
    I2_vvv -= np.transpose(I2_vvv, (1, 0, 2))

    X3 = -(2.0 / 12.0) * np.einsum("mj,abcmk->abcjk", H1[o, o], r3, optimize=True)       # (1)
    X3 += (3.0 / 12.0) * np.einsum("be,aecjk->abcjk", H1[v, v], r3, optimize=True)       # (2)
    X3 += (3.0 / 24.0) * np.einsum("abef,efcjk->abcjk", H2[v, v, v, v], r3, optimize=True)  # (3)
    X3 += (1.0 / 24.0) * np.einsum("mnjk,abcmn->abcjk", H2[o, o, o, o], r3, optimize=True)  # (4)
    X3 += (6.0 / 12.0) * np.einsum("cmke,abejm->abcjk", H2[v, o, o, v], r3, optimize=True)  # (5)
    X3 -= (3.0 / 12.0) * np.einsum("cmkj,abm->abcjk", H2[v, o, o, o], r2, optimize=True)     # (7)
    X3 += (6.0 / 12.0) * np.einsum("cbke,aej->abcjk", H2[v, v, o, v], r2, optimize=True)     # (8)
    X3 -= (6.0 / 12.0) * np.einsum("amj,bcmk->abcjk", I2_voo, t2, optimize=True) # (9)
    X3 += (3.0 / 12.0) * np.einsum("abe,ecjk->abcjk", I2_vvv, t2, optimize=True) # (10)
    X3 -= np.transpose(X3, (1, 0, 2, 3, 4)) + np.transpose(X3, (2, 1, 0, 3, 4)) # antisymmetrize A(a/bc)
    X3 -= np.transpose(X3, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3 -= np.transpose(X3, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)

    return X3

