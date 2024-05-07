import time
import numpy as np
from miniccpy.helper_cc3 import compute_leftcc3_intermediates, get_lr_intermediates, compute_eomcc3_intermediates

def kernel(R, T, omega, fock, g, H1, H2, o, v, maxit=80, convergence=1.0e-07, diis_size=6, do_diis=True):
    """
    Solve the nonlinear equations defined by the CC3 Jacobian transpose eigenvalue problem
    L*H(omega) = omega*L, where L is defined as (L1, L2). This is the left eigenvalue
    problem associated with the standard CC3 right eigenvalue problem. 
    """
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
    
    # Set the initial vector to be R
    r1, r2 = R
    L = np.hstack([r1.flatten(), r2.flatten()])

    # Allocate the DIIS engine
    if do_diis:
        out_of_core = False
        diis_engine = DIIS(ndim, diis_size, out_of_core)

    print("    ==> Left-EOM-CC3 iterations <==")
    print("    The initial guess energy = ", omega)
    print("")
    print("     Iter               Energy                 |dE|                 |dR|")
    for niter in range(maxit):
        tic = time.time()

        # Store old omega eigenvalue
        omega_old = omega

        # Normalize the L vector
        L /= np.linalg.norm(L)

        # Compute H*R for a given omega
        sigma = LH(omega, 
                   L[:n1].reshape(nunocc, nocc), L[n1:].reshape(nunocc, nunocc, nocc, nocc),
                   t1, t2, fock, g, H1, H2, o, v, e_abc)

        # Update the value of omega
        omega = np.dot(sigma.T, L)

        # Compute the eigenproblem residual L*H(omega) - omega*L
        residual = (sigma - omega * L)
        res_norm = np.linalg.norm(residual)
        delta_e = omega - omega_old

        if res_norm < convergence and abs(delta_e) < convergence:
            toc = time.time()
            minutes, seconds = divmod(toc - tic, 60)
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(niter, omega, delta_e, res_norm, minutes, seconds))
            break

        # Perturbational update step u_K = l_K/(omega-D_K), where D_K = energy denominator
        u = update(residual[:n1].reshape(nunocc, nocc),
                   residual[n1:].reshape(nunocc, nunocc, nocc, nocc),
                   omega, e_ai, e_abij)

        # Add correction vector to R
        L += u

        # Extrapolate DIIS
        if do_diis:
            diis_engine.push((L[:n1].reshape(nunocc, nocc), L[n1:].reshape(nunocc, nunocc, nocc, nocc)),
                             (u[:n1].reshape(nunocc, nocc), u[n1:].reshape(nunocc, nunocc, nocc, nocc)),
                              niter)
            if niter >= diis_size:
                L = diis_engine.extrapolate()

        # Print iteration
        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(niter, omega, delta_e, res_norm, minutes, seconds))
    else:
        print("Left-EOM-CC3 iterations did not converge")

    if do_diis:
        diis_engine.cleanup()

    # Normalize <L|R> = 1
    LR = calc_LR(L, R, t1, t2, fock, g, H1, H2, omega, e_abc, nocc, nunocc, o, v)
    L /= LR
    # Save the final converged root in an excitation tuple
    L = (L[:n1].reshape(nunocc, nocc), L[n1:].reshape(nunocc, nunocc, nocc, nocc))
    return L, omega

def calc_LR(L, R, t1, t2, f, g, H1, H2, omega, e_abc, nocc, nunocc, o, v):
    n1 = nocc*nunocc
    n2 = nocc**2 * nunocc**2
    # unpack L
    l1 = L[:n1].reshape(nunocc, nocc)
    l2 = L[n1:].reshape(nunocc, nunocc, nocc, nocc)
    # unpack R
    r1, r2 = R
    # compute LR
    LR = np.einsum("ai,ai->", l1, r1, optimize=True)
    LR += 0.25 * np.einsum("abij,abij->", l2, r2, optimize=True)
    # compute intermediates
    h_vvov, h_vooo, x_vvov, x_vooo = compute_eomcc3_intermediates(r1, r2, t1, t2, f, g, o, v)
    for i in range(nocc):
        for j in range(i + 1, nocc):
            for k in range(j + 1, nocc):
                # fock denominator for occupied
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                ### Compute R3(abc) ###
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

                ### Compute L3(abc) ###
                l3_abc = 0.5 * (
                        np.einsum("eba,ec->abc", H2[v, o, v, v][:, i, :, :], l2[:, :, j, k], optimize=True)
                        -np.einsum("eba,ec->abc", H2[v, o, v, v][:, j, :, :], l2[:, :, i, k], optimize=True)
                        -np.einsum("eba,ec->abc", H2[v, o, v, v][:, k, :, :], l2[:, :, j, i], optimize=True)
                )
                l3_abc -= 0.5 * (
                        np.einsum("ma,bcm->abc", H2[o, o, o, v][j, i, :, :], l2[:, :, :, k], optimize=True)
                        -np.einsum("ma,bcm->abc", H2[o, o, o, v][k, i, :, :], l2[:, :, :, j], optimize=True)
                        -np.einsum("ma,bcm->abc", H2[o, o, o, v][j, k, :, :], l2[:, :, :, i], optimize=True)
                )
                l3_abc += 0.5 * (
                        np.einsum("ab,c->abc", H2[o, o, v, v][i, j, :, :], l1[:, k], optimize=True)
                        -np.einsum("ab,c->abc", H2[o, o, v, v][k, j, :, :], l1[:, i], optimize=True)
                        -np.einsum("ab,c->abc", H2[o, o, v, v][i, k, :, :], l1[:, j], optimize=True)
                )
                l3_abc += 0.5 * (
                        np.einsum("a,bc->abc", H1[o, v][i, :], l2[:, :, j, k], optimize=True)
                        -np.einsum("a,bc->abc", H1[o, v][j, :], l2[:, :, i, k], optimize=True)
                        -np.einsum("a,bc->abc", H1[o, v][k, :], l2[:, :, j, i], optimize=True)
                )
                # antisymmetrize A(abc)
                l3_abc -= np.transpose(l3_abc, (1, 0, 2)) + np.transpose(l3_abc, (2, 1, 0)) # (a/bc)
                l3_abc -= np.transpose(l3_abc, (0, 2, 1)) # (bc)
                # Divide l_abc by the denominator
                l3_abc /= (omega + denom_occ + e_abc)
                LR += (1.0 / 6.0) * np.einsum("abc,abc->", r3_abc, l3_abc, optimize=True)
    return LR

def update(l1, l2, omega, e_ai, e_abij):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""
    l1 /= (omega - e_ai)
    l2 /= (omega - e_abij)
    return np.hstack([l1.flatten(), l2.flatten()])

def LH(omega, l1, l2, t1, t2, f, g, H1, H2, o, v, e_abc):
    """Compute the matrix-vector product L * H, where
    H is the CCSDT similarity-transformed Hamiltonian and L is
    the EOMCCSDT linear de-excitation operator."""
    # Get CCS intermediates (it would be nice to not have to recompute these in left-CC)
    h_vvov, h_vooo, h_voov, h_vvvv, h_oooo = compute_leftcc3_intermediates(t1, t2, f, g, o, v)
    # comptute L*T intermediates
    X1, X2 = get_lr_intermediates(l1, l2, t2, f, H1, H2, h_vvov, h_vooo, omega, e_abc, o, v)
    LH1 = LH_singles(l1, l2, t1, t2, H1, H2, X1, X2, h_voov, h_vvvv, h_oooo, o, v)
    LH2 = LH_doubles(l1, l2, t1, t2, f, H1, H2, X1, X2, h_vvov, h_vooo, omega, e_abc, o, v)
    return np.hstack( [LH1.flatten(), LH2.flatten()] )

def LH_singles(l1, l2, t1, t2, H1, H2, X1, X2, h_voov, h_vvvv, h_oooo, o, v):
    """Compute the projection of the CCSD Hamiltonian on singles
        X[a, i] = < 0 | (1 + L1 + L2)*(H_N exp(T1+T2))_C | 0 >
    """
    LH = np.einsum("ea,ei->ai", H1[v, v], l1, optimize=True)
    LH -= np.einsum("im,am->ai", H1[o, o], l1, optimize=True)
    LH += np.einsum("eima,em->ai", H2[v, o, o, v], l1, optimize=True)
    LH += 0.5 * np.einsum("fena,efin->ai", H2[v, v, o, v], l2, optimize=True)
    LH -= 0.5 * np.einsum("finm,afmn->ai", H2[v, o, o, o], l2, optimize=True)

    I1 = 0.25 * np.einsum("efmn,fgnm->ge", l2, t2, optimize=True)
    I2 = -0.25 * np.einsum("efmn,egnm->gf", l2, t2, optimize=True)
    I3 = -0.25 * np.einsum("efmo,efno->mn", l2, t2, optimize=True)
    I4 = 0.25 * np.einsum("efmo,efnm->on", l2, t2, optimize=True)
    LH += np.einsum("ge,eiga->ai", I1, H2[v, o, v, v], optimize=True)
    LH += np.einsum("gf,figa->ai", I2, H2[v, o, v, v], optimize=True)
    LH += np.einsum("mn,nima->ai", I3, H2[o, o, o, v], optimize=True)
    LH += np.einsum("on,nioa->ai", I4, H2[o, o, o, v], optimize=True)

    # < 0 | L2 * (H(2) * T3)_C | ia >
    LH += np.einsum("em,imae->ai", X1["vo"], H2[o, o, v, v], optimize=True)
    LH += 0.5 * np.einsum("nmoa,iomn->ai", X2["ooov"], h_oooo, optimize=True)
    LH += np.einsum("fmae,eimf->ai", X2["vovv"], h_voov, optimize=True)
    LH -= 0.5 * np.einsum("gife,efag->ai", X2["vovv"], h_vvvv, optimize=True)
    LH -= np.einsum("imne,enma->ai", X2["ooov"], h_voov, optimize=True)
    return LH

def LH_doubles(l1, l2, t1, t2, f, H1, H2, X1, X2, h_vvov, h_vooo, omega, e_abc, o, v):
    """Compute the projection of the CCSD Hamiltonian on doubles
        X[a, b, i, j] = < ijab | (H_N exp(T1+T2))_C | 0 >
    """
    LH = 0.5 * np.einsum("ea,ebij->abij", H1[v, v], l2, optimize=True)
    LH -= 0.5 * np.einsum("im,abmj->abij", H1[o, o], l2, optimize=True)
    LH += np.einsum("jb,ai->abij", H1[o, v], l1, optimize=True)
    I1 = (
          -0.5 * np.einsum("afmn,efmn->ea", l2, t2, optimize=True)
    )
    LH += 0.5 * np.einsum("ea,ijeb->abij", I1, H2[o, o, v, v], optimize=True)
    I1 = (
          0.5 * np.einsum("efin,efmn->im", l2, t2, optimize=True)
    )
    LH -= 0.5 * np.einsum("im,mjab->abij", I1, H2[o, o, v, v], optimize=True)
    LH += np.einsum("eima,ebmj->abij", H2[v, o, o, v], l2, optimize=True)
    LH += 0.125 * np.einsum("ijmn,abmn->abij", H2[o, o, o, o], l2, optimize=True)
    LH += 0.125 * np.einsum("efab,efij->abij", H2[v, v, v, v], l2, optimize=True)
    LH += 0.5 * np.einsum("ejab,ei->abij", H2[v, o, v, v], l1, optimize=True)
    LH -= 0.5 * np.einsum("ijmb,am->abij", H2[o, o, o, v], l1, optimize=True)

    # Moment-like terms
    nu, no = l1.shape
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                # fock denominator for occupied
                denom_occ = f[o, o][i, i] + f[o, o][j, j] + f[o, o][k, k]
                l3_abc = 0.5 * (
                        np.einsum("eba,ec->abc", H2[v, o, v, v][:, i, :, :], l2[:, :, j, k], optimize=True)
                        -np.einsum("eba,ec->abc", H2[v, o, v, v][:, j, :, :], l2[:, :, i, k], optimize=True)
                        -np.einsum("eba,ec->abc", H2[v, o, v, v][:, k, :, :], l2[:, :, j, i], optimize=True)
                )
                l3_abc -= 0.5 * (
                        np.einsum("ma,bcm->abc", H2[o, o, o, v][j, i, :, :], l2[:, :, :, k], optimize=True)
                        -np.einsum("ma,bcm->abc", H2[o, o, o, v][k, i, :, :], l2[:, :, :, j], optimize=True)
                        -np.einsum("ma,bcm->abc", H2[o, o, o, v][j, k, :, :], l2[:, :, :, i], optimize=True)
                )
                l3_abc += 0.5 * (
                        np.einsum("ab,c->abc", H2[o, o, v, v][i, j, :, :], l1[:, k], optimize=True)
                        -np.einsum("ab,c->abc", H2[o, o, v, v][k, j, :, :], l1[:, i], optimize=True)
                        -np.einsum("ab,c->abc", H2[o, o, v, v][i, k, :, :], l1[:, j], optimize=True)
                )
                l3_abc += 0.5 * (
                        np.einsum("a,bc->abc", H1[o, v][i, :], l2[:, :, j, k], optimize=True)
                        -np.einsum("a,bc->abc", H1[o, v][j, :], l2[:, :, i, k], optimize=True)
                        -np.einsum("a,bc->abc", H1[o, v][k, :], l2[:, :, j, i], optimize=True)
                )
                # antisymmetrize A(abc)
                l3_abc -= np.transpose(l3_abc, (1, 0, 2)) + np.transpose(l3_abc, (2, 1, 0)) # (a/bc)
                l3_abc -= np.transpose(l3_abc, (0, 2, 1)) # (bc)
                # Divide l_abc by the denominator
                l3_abc /= (omega + denom_occ + e_abc)
                # X2(abij) = 1/2 A(k/ij) l3(efbijk) * h_vvov(feka)
                LH[:, :, i, j] += 0.5 * np.einsum("ebf,fea->ab", l3_abc, h_vvov[:, :, k, :], optimize=True)
                LH[:, :, j, k] += 0.5 * np.einsum("ebf,fea->ab", l3_abc, h_vvov[:, :, i, :], optimize=True)
                LH[:, :, i, k] -= 0.5 * np.einsum("ebf,fea->ab", l3_abc, h_vvov[:, :, j, :], optimize=True)
                # X2(abij) = -1/2 A(j/ik) l3(abfijk) * h_vooo(f:ki)
                LH[:, :, :, j] -= 0.5 * np.einsum("abf,fi->abi", l3_abc, h_vooo[:, :, k, i], optimize=True)
                LH[:, :, :, i] += 0.5 * np.einsum("abf,fi->abi", l3_abc, h_vooo[:, :, k, j], optimize=True)
                LH[:, :, :, k] += 0.5 * np.einsum("abf,fi->abi", l3_abc, h_vooo[:, :, j, i], optimize=True)

    LH -= np.transpose(LH, (1, 0, 2, 3))
    LH -= np.transpose(LH, (0, 1, 3, 2))
    # Manually clear all diagonal elements
    for a in range(nu):
        LH[a, a, :, :] *= 0.0
    for i in range(no):
        LH[:, :, i, i] *= 0.0
    return LH
