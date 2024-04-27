import time
import numpy as np
from miniccpy.helper_cc3 import compute_eomcc3_intermediates

def kernel(R0, T, omega, fock, g, H1, H2, o, v, maxit=80, convergence=1.0e-07, diis_size=6, do_diis=True, denom_type="fock"):
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

    eps = np.diagonal(fock)
    n = np.newaxis
    # The R3 amplitudes must be defined with MP denominator in order to be consistent with CC3
    e_abc = -eps[v, n, n] - eps[n, v, n] - eps[n, n, v]
    e_abij = (eps[v, n, n, n] + eps[n, v, n, n] - eps[n, n, o, n] - eps[n, n, n, o])
    e_ai = (eps[v, n] - eps[n, o])

    # Unpack the T vectors
    t1, t2 = T

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
        out_of_core = False
        diis_engine = DIIS(ndim, diis_size, out_of_core)

    print("    ==> EOM-CC3 iterations <==")
    print("    The initial guess energy = ", omega)
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
                   t1, t2, fock, g, H1, H2, o, v, e_abc)

        # Update the value of omega
        omega = np.dot(sigma.T, R)

        # Compute the eigenproblem residual H(omega)*R - omega*R
        residual = (sigma - omega * R)
        res_norm = np.linalg.norm(residual)
        delta_e = omega - omega_old

        if res_norm < convergence and abs(delta_e) < convergence:
            toc = time.time()
            minutes, seconds = divmod(toc - tic, 60)
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(niter, omega, delta_e, res_norm, minutes, seconds))
            break

        # Perturbational update step u_K = r_K/(omega-D_K), where D_K = energy denominator
        u = update(residual[:n1].reshape(nunocc, nocc),
                   residual[n1:].reshape(nunocc, nunocc, nocc, nocc),
                   omega, e_ai, e_abij)

        # Add correction vector to R
        R += u

        # Extrapolate DIIS
        if do_diis:
            diis_engine.push((R[:n1].reshape(nunocc, nocc), R[n1:].reshape(nunocc, nunocc, nocc, nocc)),
                             (u[:n1].reshape(nunocc, nocc), u[n1:].reshape(nunocc, nunocc, nocc, nocc)),
                              niter)
            if niter >= diis_size:
                R = diis_engine.extrapolate()

        # Print iteration
        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(niter, omega, delta_e, res_norm, minutes, seconds))
    else:
        print("EOM-CC3 iterations did not converge")

    if do_diis:
        diis_engine.cleanup()

    # Save the final converged root in an excitation tuple
    R = (R[:n1].reshape(nunocc, nocc), R[n1:].reshape(nunocc, nunocc, nocc, nocc))
    # Calculate r0 for the root
    r0 = calc_r0(R[0], R[1], H1, H2, omega, o, v)
    # Compute relative excitation level diagnostic
    rel = calc_rel(r0, R[0], R[1])
    return R, omega, r0, rel

def update(r1, r2, omega, e_ai, e_abij):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""
    r1 /= (omega - e_ai)
    r2 /= (omega - e_abij)
    return np.hstack([r1.flatten(), r2.flatten()])

def HR(omega, r1, r2, t1, t2, f, g, H1, H2, o, v, e_abc):
    """Compute the matrix-vector product H * R, where
    H is the CCSDT similarity-transformed Hamiltonian and R is
    the EOMCCSDT linear excitation operator."""
    # compute intermediates
    h_vvov, h_vooo, x_vvov, x_vooo = compute_eomcc3_intermediates(r1, r2, t1, t2, f, g, o, v)
    # update R1
    HR1 = build_HR1(r1, r2, t2, omega, f, H1, H2, h_vvov, h_vooo, x_vvov, x_vooo, e_abc, o, v)
    # update R2
    HR2 = build_HR2(r1, r2, t2, omega, f, H1, H2, h_vvov, h_vooo, x_vvov, x_vooo, e_abc, o, v)
    return np.hstack( [HR1.flatten(), HR2.flatten()] )

def build_HR1(r1, r2, t2, omega, f, H1, H2, h_vvov, h_vooo, x_vvov, x_vooo, e_abc, o, v):
    """Compute the projection of HR on singles
        X[a, i] = < ia | [ HBar(CCSDT) * (R1 + R2 + R3) ]_C | 0 >
    """
    nu, no = r1.shape

    X1 = -np.einsum("mi,am->ai", H1[o, o], r1, optimize=True)
    X1 += np.einsum("ae,ei->ai", H1[v, v], r1, optimize=True)
    X1 += np.einsum("amie,em->ai", H2[v, o, o, v], r1, optimize=True)
    X1 -= 0.5 * np.einsum("mnif,afmn->ai", H2[o, o, o, v], r2, optimize=True)
    X1 += 0.5 * np.einsum("anef,efin->ai", H2[v, o, v, v], r2, optimize=True)
    X1 += np.einsum("me,aeim->ai", H1[o, v], r2, optimize=True)
    #X1 += 0.25 * np.einsum("mnef,aefimn->ai", H2[o, o, v, v], r3, optimize=True)
    # Parts contracted with R3
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                # fock denominator for occupied
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                # -1/2 A(k/ij)A(abc) X(amij) * t(bcmk)
                r3_abc = -0.5 * np.einsum("am,bcm->abc", x_vooo[:, :, i, j], t2[:, :, :, k], optimize=True)
                r3_abc += 0.5 * np.einsum("am,bcm->abc", x_vooo[:, :, k, j], t2[:, :, :, i], optimize=True)
                r3_abc += 0.5 * np.einsum("am,bcm->abc", x_vooo[:, :, i, k], t2[:, :, :, j], optimize=True)
                #
                r3_abc -= 0.5 * np.einsum("am,bcm->abc", h_vooo[:, :, i, j], r2[:, :, :, k], optimize=True)
                r3_abc += 0.5 * np.einsum("am,bcm->abc", h_vooo[:, :, k, j], r2[:, :, :, i], optimize=True)
                r3_abc += 0.5 * np.einsum("am,bcm->abc", h_vooo[:, :, i, k], r2[:, :, :, j], optimize=True)
                # 1/2 A(i/jk)A(abc) X(abie) * t(ecjk)
                r3_abc += 0.5 * np.einsum("abe,ec->abc", x_vvov[:, :, i, :], t2[:, :, j, k], optimize=True)
                r3_abc -= 0.5 * np.einsum("abe,ec->abc", x_vvov[:, :, j, :], t2[:, :, i, k], optimize=True)
                r3_abc -= 0.5 * np.einsum("abe,ec->abc", x_vvov[:, :, k, :], t2[:, :, j, i], optimize=True)
                #
                r3_abc += 0.5 * np.einsum("abe,ec->abc", h_vvov[:, :, i, :], r2[:, :, j, k], optimize=True)
                r3_abc -= 0.5 * np.einsum("abe,ec->abc", h_vvov[:, :, j, :], r2[:, :, i, k], optimize=True)
                r3_abc -= 0.5 * np.einsum("abe,ec->abc", h_vvov[:, :, k, :], r2[:, :, j, i], optimize=True)
                # Antisymmetrize A(abc)
                r3_abc -= np.transpose(r3_abc, (1, 0, 2)) + np.transpose(r3_abc, (2, 1, 0)) # A(a/bc)
                r3_abc -= np.transpose(r3_abc, (0, 2, 1)) # A(bc)
                # Divide t_abc by the denominator
                r3_abc /= (omega + denom_occ + e_abc)
                # Compute diagram: 1/2 A(i/jk) v(jkbc) * t(abcijk)
                X1[:, i] += 0.5 * np.einsum("bc,abc->a", H2[o, o, v, v][j, k, :, :], r3_abc, optimize=True)
                X1[:, j] -= 0.5 * np.einsum("bc,abc->a", H2[o, o, v, v][i, k, :, :], r3_abc, optimize=True)
                X1[:, k] -= 0.5 * np.einsum("bc,abc->a", H2[o, o, v, v][j, i, :, :], r3_abc, optimize=True)
    return X1

def build_HR2(r1, r2, t2, omega, f, H1, H2, h_vvov, h_vooo, x_vvov, x_vooo, e_abc, o, v):
    """Compute the projection of HR on doubles
        X[a, b, i, j] = < ijab | [ HBar(CCSDT) * (R1 + R2 + R3) ]_C | 0 >
    """
    nu, no = r1.shape

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
    #X2 += 0.25 * np.einsum("me,abeijm->abij", I_ov, t3, optimize=True)

    #X2 += 0.25 * np.einsum("me,abeijm->abij", H1[o, v], r3, optimize=True)
    #X2 -= 0.5 * 0.5 * np.einsum("mnjf,abfimn->abij", H2[o, o, o, v], r3, optimize=True)
    #X2 += 0.5 * 0.5 * np.einsum("bnef,aefijn->abij", H2[v, o, v, v], r3, optimize=True)

    # Parts contracted with T3/R3
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                # fock denominator for occupied
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                #### Compute R3 ####
                # -1/2 A(k/ij)A(abc) X(amij) * t(bcmk)
                r3_abc = -0.5 * np.einsum("am,bcm->abc", x_vooo[:, :, i, j], t2[:, :, :, k], optimize=True)
                r3_abc += 0.5 * np.einsum("am,bcm->abc", x_vooo[:, :, k, j], t2[:, :, :, i], optimize=True)
                r3_abc += 0.5 * np.einsum("am,bcm->abc", x_vooo[:, :, i, k], t2[:, :, :, j], optimize=True)
                #
                r3_abc -= 0.5 * np.einsum("am,bcm->abc", h_vooo[:, :, i, j], r2[:, :, :, k], optimize=True)
                r3_abc += 0.5 * np.einsum("am,bcm->abc", h_vooo[:, :, k, j], r2[:, :, :, i], optimize=True)
                r3_abc += 0.5 * np.einsum("am,bcm->abc", h_vooo[:, :, i, k], r2[:, :, :, j], optimize=True)
                # 1/2 A(i/jk)A(abc) X(abie) * t(ecjk)
                r3_abc += 0.5 * np.einsum("abe,ec->abc", x_vvov[:, :, i, :], t2[:, :, j, k], optimize=True)
                r3_abc -= 0.5 * np.einsum("abe,ec->abc", x_vvov[:, :, j, :], t2[:, :, i, k], optimize=True)
                r3_abc -= 0.5 * np.einsum("abe,ec->abc", x_vvov[:, :, k, :], t2[:, :, j, i], optimize=True)
                #
                r3_abc += 0.5 * np.einsum("abe,ec->abc", h_vvov[:, :, i, :], r2[:, :, j, k], optimize=True)
                r3_abc -= 0.5 * np.einsum("abe,ec->abc", h_vvov[:, :, j, :], r2[:, :, i, k], optimize=True)
                r3_abc -= 0.5 * np.einsum("abe,ec->abc", h_vvov[:, :, k, :], r2[:, :, j, i], optimize=True)
                # Antisymmetrize A(abc)
                r3_abc -= np.transpose(r3_abc, (1, 0, 2)) + np.transpose(r3_abc, (2, 1, 0)) # A(a/bc)
                r3_abc -= np.transpose(r3_abc, (0, 2, 1)) # A(bc)
                # Divide t_abc by the denominator
                r3_abc /= (omega + denom_occ + e_abc)

                #### Compute T3 ####
                # -1/2 A(k/ij)A(abc) X(amij) * t(bcmk)
                t3_abc = -0.5 * np.einsum("am,bcm->abc", h_vooo[:, :, i, j], t2[:, :, :, k], optimize=True)
                t3_abc += 0.5 * np.einsum("am,bcm->abc", h_vooo[:, :, k, j], t2[:, :, :, i], optimize=True)
                t3_abc += 0.5 * np.einsum("am,bcm->abc", h_vooo[:, :, i, k], t2[:, :, :, j], optimize=True)
                # 1/2 A(i/jk)A(abc) X(abie) * t(ecjk)
                t3_abc += 0.5 * np.einsum("abe,ec->abc", h_vvov[:, :, i, :], t2[:, :, j, k], optimize=True)
                t3_abc -= 0.5 * np.einsum("abe,ec->abc", h_vvov[:, :, j, :], t2[:, :, i, k], optimize=True)
                t3_abc -= 0.5 * np.einsum("abe,ec->abc", h_vvov[:, :, k, :], t2[:, :, j, i], optimize=True)
                # Antisymmetrize A(abc)
                t3_abc -= np.transpose(t3_abc, (1, 0, 2)) + np.transpose(t3_abc, (2, 1, 0)) # A(a/bc)
                t3_abc -= np.transpose(t3_abc, (0, 2, 1)) # A(bc)
                # Divide t_abc by the denominator
                t3_abc /= (denom_occ + e_abc)

                # Compute diagram: A(ij) [A(k/ij) h(ke) * t3(abeijk)]
                X2[:, :, i, j] += 0.5 * np.einsum("e,abe->ab", I_ov[k, :], t3_abc, optimize=True) # (1)
                X2[:, :, j, k] += 0.5 * np.einsum("e,abe->ab", I_ov[i, :], t3_abc, optimize=True) # (ik)
                X2[:, :, i, k] -= 0.5 * np.einsum("e,abe->ab", I_ov[j, :], t3_abc, optimize=True) # (jk)
                # Compute diagram: A(ij) [A(k/ij) h(ke) * t3(abeijk)]
                X2[:, :, i, j] += 0.5 * np.einsum("e,abe->ab", H1[o, v][k, :], r3_abc, optimize=True) # (1)
                X2[:, :, j, k] += 0.5 * np.einsum("e,abe->ab", H1[o, v][i, :], r3_abc, optimize=True) # (ik)
                X2[:, :, i, k] -= 0.5 * np.einsum("e,abe->ab", H1[o, v][j, :], r3_abc, optimize=True) # (jk)
                # Compute diagram: -A(j/ik) h(ik:f) * t3(abfijk)
                X2[:, :, :, j] -= 0.5 * np.einsum("mf,abf->abm", H2[o, o, o, v][i, k, :, :], r3_abc, optimize=True) 
                X2[:, :, :, i] += 0.5 * np.einsum("mf,abf->abm", H2[o, o, o, v][j, k, :, :], r3_abc, optimize=True) 
                X2[:, :, :, k] += 0.5 * np.einsum("mf,abf->abm", H2[o, o, o, v][i, j, :, :], r3_abc, optimize=True) 
                # Compute diagram: 1/2 A(k/ij) h(akef) * t3(ebfijk) 
                X2[:, :, i, j] += 0.5 * np.einsum("aef,ebf->ab", H2[v, o, v, v][:, k, :, :], r3_abc, optimize=True) 
                X2[:, :, j, k] += 0.5 * np.einsum("aef,ebf->ab", H2[v, o, v, v][:, i, :, :], r3_abc, optimize=True) 
                X2[:, :, i, k] -= 0.5 * np.einsum("aef,ebf->ab", H2[v, o, v, v][:, j, :, :], r3_abc, optimize=True) 

    # Antisymmetrize
    X2 -= np.transpose(X2, (0, 1, 3, 2))
    X2 -= np.transpose(X2, (1, 0, 2, 3))
    # Manually clear all diagonal elements
    for a in range(nu):
        X2[a, a, :, :] *= 0.0
    for i in range(no):
        X2[:, :, i, i] *= 0.0
    return X2

