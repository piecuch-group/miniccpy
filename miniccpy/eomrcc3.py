import time
import numpy as np
from miniccpy.utilities import get_memory_usage
from miniccpy.helper_cc3 import compute_rccs_intermediates, compute_eomrcc3_intermediates


def kernel(R0, T, omega, fock, g, H1, H2, o, v, maxit=80, convergence=1.0e-07, diis_size=6, do_diis=True):
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
    from miniccpy.energy import calc_r0_rhf, calc_rel_rhf
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
    n2 = nocc ** 2 * nunocc ** 2
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

    # Compute CCS Hbar intermediates
    h_vvov, h_vooo, h_voov, h_vovo, h_vvvv, h_oooo = compute_rccs_intermediates(t1, t2, fock, g, o, v)

    print("    ==> EOM-RCC3 iterations <==")
    print("    The initial guess energy = ", omega)
    print("")
    print("     Iter               Energy                 |dE|                 |dR|     Wall Time     Memory")
    for niter in range(maxit):
        tic = time.time()

        # Store old omega eigenvalue
        omega_old = omega

        # Normalize the R vector
        R /= np.linalg.norm(R)

        # Compute H*R for a given omega
        sigma = HR(omega,
                   R[:n1].reshape(nunocc, nocc), R[n1:].reshape(nunocc, nunocc, nocc, nocc),
                   t1, t2, fock, g, H1, H2,
                   h_vvov, h_vooo, h_voov, h_vovo, h_vvvv, h_oooo,
                   o, v, e_abc)

        # Update the value of omega
        omega = np.dot(sigma.T, R)

        # Compute the eigenproblem residual H(omega)*R - omega*R
        residual = (sigma - omega * R)
        res_norm = np.linalg.norm(residual)
        delta_e = omega - omega_old

        if res_norm < convergence and abs(delta_e) < convergence:
            toc = time.time()
            minutes, seconds = divmod(toc - tic, 60)
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega,
                                                                                                       delta_e,
                                                                                                       res_norm,
                                                                                                       minutes, seconds,
                                                                                                       get_memory_usage()))
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
        print(
            "    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e,
                                                                                                 res_norm, minutes,
                                                                                                 seconds,
                                                                                                 get_memory_usage()))
    else:
        print("EOM-CC3 iterations did not converge")

    if do_diis:
        diis_engine.cleanup()

    # Save the final converged root in an excitation tuple
    R = (R[:n1].reshape(nunocc, nocc), R[n1:].reshape(nunocc, nunocc, nocc, nocc))
    # Calculate r0 for the root
    r0 = calc_r0_rhf(R[0], R[1], H1, H2, omega, o, v)
    # Compute relative excitation level diagnostic
    rel = calc_rel_rhf(r0, R[0], R[1])
    return R, omega, r0, rel


def update(r1, r2, omega, e_ai, e_abij):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""
    r1 /= (omega - e_ai)
    r2 /= (omega - e_abij)
    return np.hstack([r1.flatten(), r2.flatten()])


def HR(omega, r1, r2, t1, t2, f, g, H1, H2,
       h_vvov, h_vooo, h_voov, h_vovo, h_vvvv, h_oooo,
       o, v, e_abc):
    """Compute the matrix-vector product H * R, where
    H is the CCSDT similarity-transformed Hamiltonian and R is
    the EOMCCSDT linear excitation operator."""
    # compute intermediates
    x_vvov, x_vooo = compute_eomrcc3_intermediates(r1, r2, h_oooo, h_voov, h_vovo, h_vvvv)
    # Add R3 parts
    HR1, HR2 = add_r3_contributions(r1, r2, t1, t2, omega, f, g, H1, H2, h_vvov, h_vooo, x_vvov, x_vooo, e_abc, o, v)
    # Add T3 parts
    HR2 += add_t3_contributions(r1, t2, f, g, h_vooo, h_vvov, e_abc, o, v)
    # update R1
    HR1 += build_HR1(r1, r2, H1, H2, o, v)
    # update R2
    HR2 += build_HR2(r1, r2, t1, t2, H1, H2, o, v)
    return np.hstack([HR1.flatten(), HR2.flatten()])

def build_HR1(r1, r2, H1, H2, o, v):
    """Compute the projection of HR on singles
        X[a, i] = < ia | [ HBar(CCSD) * (R1 + R2) ]_C | 0 >
    """
    X1 = -np.einsum("mi,am->ai", H1[o, o], r1, optimize=True)
    X1 += np.einsum("ae,ei->ai", H1[v, v], r1, optimize=True)
    X1 += 2.0 * np.einsum("me,aeim->ai", H1[o, v], r2, optimize=True)
    X1 -= np.einsum("me,aemi->ai", H1[o, v], r2, optimize=True)
    X1 += 2.0 * np.einsum("amie,em->ai", H2[v, o, o, v], r1, optimize=True)
    X1 -= np.einsum("amei,em->ai", H2[v, o, v, o], r1, optimize=True)
    X1 -= 2.0 * np.einsum("mnif,afmn->ai", H2[o, o, o, v], r2, optimize=True)
    X1 += np.einsum("nmif,afmn->ai", H2[o, o, o, v], r2, optimize=True)
    X1 += 2.0 * np.einsum("anef,efin->ai", H2[v, o, v, v], r2, optimize=True)
    X1 -= np.einsum("anfe,efin->ai", H2[v, o, v, v], r2, optimize=True)
    return X1

def build_HR2(r1, r2, t1, t2, H1, H2, o, v):
    """Compute the projection of HR on doubles
        X[a, b, i, j] = < ijab | [ HBar(CCSD) * (R1 + R2) ]_C | 0 >
    """
    # intermediates
    X_oo = (
            + 2.0 * np.einsum("mnjf,fn->mj", H2[o, o, o, v], r1, optimize=True)
            - np.einsum("nmjf,fn->mj", H2[o, o, o, v], r1, optimize=True)
            + 2.0 * np.einsum("mnef,efjn->mj", H2[o, o, v, v], r2, optimize=True)
            - np.einsum("nmef,efjn->mj", H2[o, o, v, v], r2, optimize=True)
    )
    X_vv = (
            + 2.0 * np.einsum("bnef,fn->be", H2[v, o, v, v], r1, optimize=True)
            - np.einsum("bnfe,fn->be", H2[v, o, v, v], r1, optimize=True)
            - 2.0 * np.einsum("mnef,bfmn->be", H2[o, o, v, v], r2, optimize=True)
            + np.einsum("nmef,bfmn->be", H2[o, o, v, v], r2, optimize=True)
    )
    # < IJAB | (H(2)*(R1+R2))_C | 0 >
    X2 = np.einsum("ae,ebij->abij", H1[v, v], r2, optimize=True)
    X2 -= np.einsum("mi,abmj->abij", H1[o, o], r2, optimize=True)
    X2 += 0.5 * np.einsum("mnij,abmn->abij", H2[o, o, o, o], r2, optimize=True)
    X2 += 0.5 * np.einsum("abef,efij->abij", H2[v, v, v, v], r2, optimize=True)
    X2 += np.einsum("baje,ei->abij", H2[v, v, o, v], r1, optimize=True)
    X2 -= np.einsum("bmji,am->abij", H2[v, o, o, o], r1, optimize=True)
    X2 += np.einsum("ae,ebij->abij", X_vv, t2, optimize=True)
    X2 -= np.einsum("mi,abmj->abij", X_oo, t2, optimize=True)
    X2 += 2.0 * np.einsum("amie,ebmj->abij", H2[v, o, o, v], r2, optimize=True)
    X2 -= np.einsum("amie,ebjm->abij", H2[v, o, o, v], r2, optimize=True)
    X2 -= np.einsum("amei,ebmj->abij", H2[v, o, v, o], r2, optimize=True)
    X2 -= np.einsum("amej,ebim->abij", H2[v, o, v, o], r2, optimize=True)
    X2 += X2.transpose(1, 0, 3, 2)
    return X2

def add_r3_contributions(r1, r2, t1, t2, omega, f, g, H1, H2, h_vvov, h_vooo, x_vvov, x_vooo, e_abc, o, v):
    nu, no = t1.shape
    # RHF-adapted integral element
    gs_oovv = 2.0 * g[o, o, v, v] - g[o, o, v, v].swapaxes(2, 3)
    # residual containers
    X1 = np.zeros((nu, no))
    X2 = np.zeros((nu, nu, no, no))
    for i in range(no):
        for j in range(no):
            for k in range(no):
                if i == j and j == k: continue
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                # -h2(amij) * r2(bcmk)
                m3 = -np.einsum("am,bcm->abc", h_vooo[:, :, i, j], r2[:, :, :, k], optimize=True) # (1)
                m3 -= np.einsum("bm,acm->abc", h_vooo[:, :, j, i], r2[:, :, :, k], optimize=True) # (ij)(ab)
                m3 -= np.einsum("cm,bam->abc", h_vooo[:, :, k, j], r2[:, :, :, i], optimize=True) # (ac)(ik)
                m3 -= np.einsum("am,cbm->abc", h_vooo[:, :, i, k], r2[:, :, :, j], optimize=True) # (bc)(jk)
                m3 -= np.einsum("bm,cam->abc", h_vooo[:, :, j, k], r2[:, :, :, i], optimize=True) # (ij)(ab)(ac)(ik)
                m3 -= np.einsum("cm,abm->abc", h_vooo[:, :, k, i], r2[:, :, :, j], optimize=True) # (ij)(ab)(bc)(jk)
                # -x2(amij) * t2(bcmk)
                m3 -= np.einsum("am,bcm->abc", x_vooo[:, :, i, j], t2[:, :, :, k], optimize=True) # (1)
                m3 -= np.einsum("bm,acm->abc", x_vooo[:, :, j, i], t2[:, :, :, k], optimize=True) # (ij)(ab)
                m3 -= np.einsum("cm,bam->abc", x_vooo[:, :, k, j], t2[:, :, :, i], optimize=True) # (ac)(ik)
                m3 -= np.einsum("am,cbm->abc", x_vooo[:, :, i, k], t2[:, :, :, j], optimize=True) # (bc)(jk)
                m3 -= np.einsum("bm,cam->abc", x_vooo[:, :, j, k], t2[:, :, :, i], optimize=True) # (ij)(ab)(ac)(ik)
                m3 -= np.einsum("cm,abm->abc", x_vooo[:, :, k, i], t2[:, :, :, j], optimize=True) # (ij)(ab)(bc)(jk)
                # h2(abie) * r2(bcek)
                m3 += np.einsum("abe,ec->abc", h_vvov[:, :, i, :], r2[:, :, j, k], optimize=True) # (1)
                m3 += np.einsum("bae,ec->abc", h_vvov[:, :, j, :], r2[:, :, i, k], optimize=True) # (ij)(ab)
                m3 += np.einsum("cbe,ea->abc", h_vvov[:, :, k, :], r2[:, :, j, i], optimize=True) # (ac)(ik)
                m3 += np.einsum("ace,eb->abc", h_vvov[:, :, i, :], r2[:, :, k, j], optimize=True) # (bc)(jk)
                m3 += np.einsum("bce,ea->abc", h_vvov[:, :, j, :], r2[:, :, k, i], optimize=True) # (ij)(ab)(ac)(ik)
                m3 += np.einsum("cae,eb->abc", h_vvov[:, :, k, :], r2[:, :, i, j], optimize=True) # (ij)(ab)(bc)(jk)
                # x2(abie) * t2(bcek)
                m3 += np.einsum("abe,ec->abc", x_vvov[:, :, i, :], t2[:, :, j, k], optimize=True) # (1)
                m3 += np.einsum("bae,ec->abc", x_vvov[:, :, j, :], t2[:, :, i, k], optimize=True) # (ij)(ab)
                m3 += np.einsum("cbe,ea->abc", x_vvov[:, :, k, :], t2[:, :, j, i], optimize=True) # (ac)(ik)
                m3 += np.einsum("ace,eb->abc", x_vvov[:, :, i, :], t2[:, :, k, j], optimize=True) # (bc)(jk)
                m3 += np.einsum("bce,ea->abc", x_vvov[:, :, j, :], t2[:, :, k, i], optimize=True) # (ij)(ab)(ac)(ik)
                m3 += np.einsum("cae,eb->abc", x_vvov[:, :, k, :], t2[:, :, i, j], optimize=True) # (ij)(ab)(bc)(jk)
                # divide by MP denominator
                m3 /= (omega + e_abc + denom_occ)
                # zero out diagonal elements
                for a in range(nu):
                    m3[a, a, a] *= 0.0
                # update singles residual
                X1[:, i] += np.einsum('abc,bc->a', m3 - m3.swapaxes(0, 2), gs_oovv[j, k, :, :], optimize=True)
                # symmetrize
                m3 = (2.0 * m3
                      - m3.swapaxes(1, 2)
                      - m3.swapaxes(0, 2)
                )
                # update doubles residual
                X2[:, :, i, j] += 0.5 * np.einsum('abc,c->ab', m3, H1[o, v][k, :])
                X2[:, :, i, j] += np.einsum('abc,dbc->ad', m3, H2[v, o, v, v][:, k, :, :])
                X2[:, :, i, :] -= np.einsum('abc,lc->abl', m3, H2[o, o, o, v][j, k, :, :])
    # Apply (ij)(ab) symmetrizer
    X2 += X2.transpose(1, 0, 3, 2)
    return X1, X2

def add_t3_contributions(r1, t2, f, g, I_vooo, I_vvov, e_abc, o, v):
    nu, no = r1.shape
    # Intermediates
    I_ov = (
          2.0 * np.einsum("mnef,fn->me", g[o, o, v, v], r1, optimize=True)
        - np.einsum("nmef,fn->me", g[o, o, v, v], r1, optimize=True)
    )
    # residual containers
    X2 = np.zeros((nu, nu, no, no))
    for i in range(no):
        for j in range(no):
            for k in range(no):
                if i == j and j == k: continue
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                # -h2(amij) * t2(bcmk)
                m3 = -np.einsum("am,bcm->abc", I_vooo[:, :, i, j], t2[:, :, :, k], optimize=True) # (1)
                m3 -= np.einsum("bm,acm->abc", I_vooo[:, :, j, i], t2[:, :, :, k], optimize=True) # (ij)(ab)
                m3 -= np.einsum("cm,bam->abc", I_vooo[:, :, k, j], t2[:, :, :, i], optimize=True) # (ac)(ik)
                m3 -= np.einsum("am,cbm->abc", I_vooo[:, :, i, k], t2[:, :, :, j], optimize=True) # (bc)(jk)
                m3 -= np.einsum("bm,cam->abc", I_vooo[:, :, j, k], t2[:, :, :, i], optimize=True) # (ij)(ab)(ac)(ik)
                m3 -= np.einsum("cm,abm->abc", I_vooo[:, :, k, i], t2[:, :, :, j], optimize=True) # (ij)(ab)(bc)(jk)
                # h2(abie) * t2(bcek)
                m3 += np.einsum("abe,ec->abc", I_vvov[:, :, i, :], t2[:, :, j, k], optimize=True) # (1)
                m3 += np.einsum("bae,ec->abc", I_vvov[:, :, j, :], t2[:, :, i, k], optimize=True) # (ij)(ab)
                m3 += np.einsum("cbe,ea->abc", I_vvov[:, :, k, :], t2[:, :, j, i], optimize=True) # (ac)(ik)
                m3 += np.einsum("ace,eb->abc", I_vvov[:, :, i, :], t2[:, :, k, j], optimize=True) # (bc)(jk)
                m3 += np.einsum("bce,ea->abc", I_vvov[:, :, j, :], t2[:, :, k, i], optimize=True) # (ij)(ab)(ac)(ik)
                m3 += np.einsum("cae,eb->abc", I_vvov[:, :, k, :], t2[:, :, i, j], optimize=True) # (ij)(ab)(bc)(jk)
                # divide by MP denominator
                m3 /= (e_abc + denom_occ)
                # zero out diagonal elements
                for a in range(nu):
                    m3[a, a, a] *= 0.0
                # symmetrize
                m3 = (2.0 * m3
                      - m3.swapaxes(1, 2)
                      - m3.swapaxes(0, 2)
                )
                # update doubles residual
                X2[:, :, i, j] += 0.5 * np.einsum('abc,c->ab', m3, I_ov[k, :])
    # Apply (ij)(ab) symmetrizer
    X2 += X2.transpose(1, 0, 3, 2)
    return X2
