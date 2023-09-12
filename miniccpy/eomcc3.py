import time
import numpy as np

def kernel(R0, T, omega, fock, g, H1, H2, o, v, maxit=80, convergence=1.0e-07, max_size=20, diis_size=6, nrest=1, do_diis=False):
    """
    Solve the nonlinear equations defined by the CC3 Jacobian eigenvalue problem
    H(omega)*R = omega*R, where R is defined as (R1, R2). This corresponds to the
    partitioned, or folded, eigenvalue problem, where the R3 operator is given by
    R3 = <ijkabc|(U*T2)_C+(H*R2)_C|0>/(omega - D_{abcijk}), with U = (H(1)*R1)_C,
    H = H(1), and D_{abcijk} = f(a,a)+f(b,b)+f(c,c)-f(i,i)-f(j,j)-f(k,k), and 
    this expression is inserted directly into the equations for R1 and R2, resulting
    in an omega-dependent Hamiltonian operator that we need to diagonalize.

    Reference: J. Chem. Phys. 113, 5154 (2000) 
    """
    from miniccpy.energy import calc_r0, calc_rel
    from miniccpy.diis import DIIS

    f_eps = np.diagonal(fock)
    n = np.newaxis
    # The R3 amplitudes must be defined with MP denominator in order to be consistent with CC3
    e_abcijk = (f_eps[v, n, n, n, n, n] + f_eps[n, v, n, n, n, n] + f_eps[n, n, v, n, n, n]
                - f_eps[n, n, n, o, n, n] - f_eps[n, n, n, n, o, n] - f_eps[n, n, n, n, n, o])
    # From my testing, using MP denominator works better than renormalized denominator with H1
    e_abij = (f_eps[v, n, n, n] + f_eps[n, v, n, n] - f_eps[n, n, o, n] - f_eps[n, n, n, o])
    e_ai = (f_eps[v, n] - f_eps[n, o])

    t1, t2, t3 = T

    nunocc, nocc = e_ai.shape
    n1 = nunocc * nocc
    n2 = nocc**2 * nunocc**2
    ndim = n1 + n2
    
    if len(R0) < ndim:
        R = np.zeros(ndim)
        R[:len(R0)] = R0
    else:
        R = R0.copy()

    # Allocate the DIIS engine
    if do_diis:
        diis_engine = DIIS(ndim, diis_size, False)

    print("    ==> EOM-CC3 iterations <==")
    print("     The initial guess energy = ", omega)
    print("")
    print("     Iter               Energy                 |dE|                 |dR|")
    for niter in range(maxit):
        tic = time.time()

        # Store old omega eigenvalue
        omega_old = omega

        # Normalize the R vector
        R /= np.linalg.norm(R)

        # Compute H*R for a given omega
        sigma = HR(omega, 
                   R[:n1].reshape(nunocc, nocc), R[n1:].reshape(nunocc, nunocc, nocc, nocc),
                   t1, t2, t3, g, H1, H2, o, v, e_abcijk)

        # Update the value of omega
        #omega = np.dot(sigma.T, R) / np.dot(R.T, R)
        omega = np.dot(sigma.T, R)

        # Compute the eigenproblem residual H(omega)*R - omega*R
        #residual = (sigma - omega * R)# / np.linalg.norm(R)
        residual = (sigma - omega * R)
        res_norm = np.linalg.norm(residual)
        delta_e = omega - omega_old

        if res_norm < convergence and abs(delta_e) < convergence:
            toc = time.time()
            minutes, seconds = divmod(toc - tic, 60)
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(niter, omega, delta_e, res_norm, minutes, seconds))
            break

        # Perturbational update step u_K = r_K/(-D_K), where D_K = energy denominator
        u = update(residual[:n1].reshape(nunocc, nocc),
                   residual[n1:].reshape(nunocc, nunocc, nocc, nocc),
                   0.0, e_ai, e_abij)
        if do_diis:
            if niter >= 3:
                diis_engine.push((u[:n1].reshape(nunocc, nocc), u[n1:].reshape(nunocc, nunocc, nocc, nocc)),
                                 (residual[:n1].reshape(nunocc, nocc), residual[n1:].reshape(nunocc, nunocc, nocc, nocc)),
                                 niter)
            if niter >= diis_size + 3:
                u = diis_engine.extrapolate()

        # Add correction vector to R
        #R = (R + u) / np.linalg.norm(R)
        R += u

        # Print iteration
        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(niter, omega, delta_e, res_norm, minutes, seconds))
    else:
        print("EOM-CC3 iterations did not converge")

    if do_diis:
        diis_engine.cleanup()

    # Calculate r0 for the root
    r0 = calc_r0(R[:n1].reshape(nunocc, nocc),
                 R[n1:].reshape(nunocc, nunocc, nocc, nocc),
                 H1, H2, omega, o, v)
    # Compute relative excitation level diagnostic
    rel = calc_rel(r0,
                   R[:n1].reshape(nunocc, nocc),
                   R[n1:].reshape(nunocc, nunocc, nocc, nocc))

    return R, omega, r0, rel

def update(r1, r2, omega, e_ai, e_abij):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""

    r1 /= (omega - e_ai)
    r2 /= (omega - e_abij)

    return np.hstack([r1.flatten(), r2.flatten()])

def HR(omega, r1, r2, t1, t2, t3, g, H1, H2, o, v, e_abcijk):
    """Compute the matrix-vector product H * R, where
    H is the CCSDT similarity-transformed Hamiltonian and R is
    the EOMCCSDT linear excitation operator."""

    # Compute the new R3
    r3 = compute_r3(r1, r2, t1, t2, omega, e_abcijk, g, o, v)
    # update R1
    HR1 = build_HR1(r1, r2, r3, H1, H2, o, v)
    # update R2
    HR2 = build_HR2(r1, r2, r3, t1, t2, t3, H1, H2, o, v)

    return np.hstack( [HR1.flatten(), HR2.flatten()] )

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
    # T3 is included in here!
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

def compute_r3(r1, r2, t1, t2, omega, e_abcijk, g, o, v):
    # CCS Hbar elements
    h_voov = g[v, o, o, v] + (
            -np.einsum("nmie,an->amie", g[o, o, o, v], t1, optimize=True)
            +np.einsum("amfe,fi->amie", g[v, o, v, v], t1, optimize=True)
            -np.einsum("mnef,an,fi->amie", g[o, o, v, v], t1, t1, optimize=True)
    )
    h_oooo = 0.5 * g[o, o, o, o] + (
            +np.einsum("nmjf,fi->mnij", g[o, o, o, v], t1, optimize=True)
            +0.5 * np.einsum("mnef,ei,fj->mnij", g[o, o, v, v], t1, t1, optimize=True)
    )
    h_oooo -= np.transpose(h_oooo, (0, 1, 3, 2))
    h_vvvv = 0.5 * g[v, v, v, v] + (
            -np.einsum("anef,bn->abef", g[v, o, v, v], t1, optimize=True)
            +0.5 * np.einsum("mnef,am,bn->abef", g[o, o, v, v,], t1, t1, optimize=True)
    )
    h_vvvv -= np.transpose(h_vvvv, (1, 0, 2, 3))
    h_vooo = 0.5 * g[v, o, o, o] + (
            +np.einsum("amie,ej->amij", g[v, o, o, v], t1, optimize=True)
            -0.5 * np.einsum("nmij,an->amij", g[o, o, o, o], t1, optimize=True)
            -np.einsum("mnjf,an,fi->amij", g[o, o, o, v], t1, t1, optimize=True)
            +0.5 * np.einsum("amef,ei,fj->amij", g[v, o, v, v], t1, t1, optimize=True)
            -0.5 * np.einsum("mnef,an,fi,ej->amij", g[o, o, v, v], t1, t1, t1, optimize=True)
    )
    h_vooo -= np.transpose(h_vooo, (0, 1, 3, 2))
    h_vvov = 0.5 * g[v, v, o, v] + (
            -np.einsum("anie,bn->abie", g[v, o, o, v], t1, optimize=True)
            +0.5 * np.einsum("abfe,fi->abie", g[v, v, v, v], t1, optimize=True)
            -np.einsum("bnef,an,fi->abie", g[v, o, v, v], t1, t1, optimize=True)
            +0.5 * np.einsum("mnie,bn,am->abie", g[o, o, o, v], t1, t1, optimize=True)
            +0.5 * np.einsum("mnef,bm,an,fi->abie", g[o, o, v, v], t1, t1, t1, optimize=True)
    )
    h_vvov -= np.transpose(h_vvov, (1, 0, 2, 3))

    X_vooo = (
                np.einsum("amie,ej->amij", h_voov, r1, optimize=True)
               -0.5 * np.einsum("mnji,an->amij", h_oooo, r1, optimize=True)
    )
    X_vooo -= np.transpose(X_vooo, (0, 1, 3, 2))
    X_vvov = (
                -np.einsum("amie,bm->abie", h_voov, r1, optimize=True)
                +0.5 * np.einsum("abfe,fi->abie", h_vvvv, r1, optimize=True)
    )
    X_vvov -= np.transpose(X_vvov, (1, 0, 2, 3))

    # <ijkabc| [H(1)*T2*R1]_C | 0 >
    X3 = 0.25 * np.einsum("baje,ecik->abcijk", X_vvov, t2, optimize=True)
    X3 -= 0.25 * np.einsum("bmji,acmk->abcijk", X_vooo, t2, optimize=True)
    # <ijkabc| [H(1)*R2]_C | 0 >
    X3 += 0.25 * np.einsum("baje,ecik->abcijk", h_vvov, r2, optimize=True)
    X3 -= 0.25 * np.einsum("bmji,acmk->abcijk", h_vooo, r2, optimize=True)
    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    X3 -= np.transpose(X3, (0, 1, 2, 3, 5, 4))
    X3 -= np.transpose(X3, (0, 1, 2, 4, 3, 5)) + np.transpose(X3, (0, 1, 2, 5, 4, 3))
    X3 -= np.transpose(X3, (0, 2, 1, 3, 4, 5))
    X3 -= np.transpose(X3, (1, 0, 2, 3, 4, 5)) + np.transpose(X3, (2, 1, 0, 3, 4, 5))

    return X3 / (omega - e_abcijk)
