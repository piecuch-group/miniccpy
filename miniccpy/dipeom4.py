import time
import numpy as np
from miniccpy.utilities import get_memory_usage

def kernel(R0, T, omega, H1, H2, o, v, maxit=80, convergence=1.0e-07, max_size=20, nrest=1):
    """
    Diagonalize the similarity-transformed CCSD Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
    from miniccpy.energy import calc_rel_dip

    eps = np.diagonal(H1)
    n = np.newaxis
    e_ijcdkl = (-eps[o, n, n, n, n, n] - eps[n, o, n, n, n, n] + eps[n, n, v, n, n, n] + eps[n, n, n, v, n, n] - eps[n, n, n, n, o, n] - eps[n, n, n, n, n, o])
    e_ijck = (-eps[o, n, n, n] - eps[n, o, n, n] + eps[n, n, v, n] - eps[n, n, n, o])
    e_ij = (-eps[o, n] - eps[n, o])

    t1, t2 = T

    nunocc, nocc = t1.shape
    n1 = nocc**2
    n2 = nocc**3 * nunocc
    n3 = nocc**4 * nunocc**2
    ndim = n1 + n2 + n3
    
    if len(R0) <= ndim:
        R = np.zeros(ndim)
        R[:len(R0)] = R0

    # Allocate the B and sigma matrices
    sigma = np.zeros((ndim, max_size))
    B = np.zeros((ndim, max_size))
    restart_block = np.zeros((ndim, nrest))

    # Initial values
    B[:, 0] = R
    sigma[:, 0] = HR(R[:n1].reshape(nocc, nocc),
                     R[n1:n1+n2].reshape(nocc, nocc, nunocc, nocc),
                     R[n1+n2:].reshape(nocc, nocc, nunocc, nunocc, nocc, nocc),
                     t1, t2, H1, H2, o, v)

    print("    ==> DIP-EOMCC(4h-2p) iterations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dR|     Wall Time     Memory")
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

        if res_norm < convergence and abs(delta_e) < convergence:
            toc = time.time()
            minutes, seconds = divmod(toc - tic, 60)
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
            break

        # update residual vector
        q = update(residual[:n1].reshape(nocc, nocc),
                   residual[n1:n1+n2].reshape(nocc, nocc, nunocc, nocc),
                   residual[n1+n2:].reshape(nocc, nocc, nunocc, nunocc, nocc, nocc),
                   omega,
                   e_ij,
                   e_ijck,
                   e_ijcdkl)
        for p in range(curr_size):
            b = B[:, p] / np.linalg.norm(B[:, p])
            q -= np.dot(b.T, q) * b
        q *= 1.0 / np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[:, curr_size] = q
            sigma[:, curr_size] = HR(q[:n1].reshape(nocc, nocc),
                                     q[n1:n1+n2].reshape(nocc, nocc, nunocc, nocc),
                                     q[n1+n2:].reshape(nocc, nocc, nunocc, nunocc, nocc, nocc),
                                     t1, t2, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[:, j] = restart_block[:, j]
                sigma[:, j] = HR(restart_block[:n1, j].reshape(nocc, nocc),
                                 restart_block[n1:n1+n2, j].reshape(nocc, nocc, nunocc, nocc),
                                 restart_block[n1+n2:, j].reshape(nocc, nocc, nunocc, nunocc, nocc, nocc),
                                 t1, t2, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        print("DIP-EOMCC(4h-2p) iterations did not converge")

    # Save the final converged root in an excitation tuple
    R = (R[:n1].reshape(nocc, nocc), R[n1:n1+n2].reshape(nocc, nocc, nunocc, nocc), R[n1+n2:].reshape(nocc, nocc, nunocc, nunocc, nocc, nocc))
    # r0 for a root in DIP is 0 by definition
    r0 = 0.0
    # Compute relative excitation level diagnostic
    rel = calc_rel_dip(R[0], R[1])
    return R, omega, r0, rel

def update(r1, r2, r3, omega, e_ij, e_ijck, e_ijcdkl):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""

    r1 /= (omega - e_ij)
    r2 /= (omega - e_ijck)
    r3 /= (omega - e_ijcdkl)

    return np.hstack([r1.flatten(), r2.flatten(), r3.flatten()])

def HR(r1, r2, r3, t1, t2, H1, H2, o, v):
    """Compute the matrix-vector product H * R, where
    H is the CCSD similarity-transformed Hamiltonian and R is
    the DIP-EOMCC linear excitation operator."""

    # update R1
    HR1 = build_HR1(r1, r2, r3, H1, H2, o, v)
    # update R2
    HR2 = build_HR2(r1, r2, r3, t1, t2, H1, H2, o, v)
    # update R3
    HR3 = build_HR3(r1, r2, r3, t1, t2, H1, H2, o, v)

    return np.hstack( [HR1.flatten(), HR2.flatten(), HR3.flatten()] )

def build_HR1(r1, r2, r3, H1, H2, o, v):
    """Compute the projection of HR on 2h excitations
        X[i, j] = < ij | [ HBar(CCSD) * (R1 + R2 + R3) ]_C | 0 >
    """
    X1 = -np.einsum("mi,mj->ij", H1[o, o], r1, optimize=True)
    X1 += 0.25 * np.einsum("mnij,mn->ij", H2[o, o, o, o], r1, optimize=True)
    X1 += 0.5 * np.einsum("me,ijem->ij", H1[o, v], r2, optimize=True)
    X1 -= 0.5 * np.einsum("mnif,mjfn->ij", H2[o, o, o, v], r2, optimize=True)
    # terms contracted with R(4p-2h)
    X1 += 0.125 * np.einsum("mnef,ijefmn->ij", H2[o, o, v, v], r3, optimize=True)
    # antisymmetrize A(ij)
    X1 -= np.transpose(X1, (1, 0))
    return X1

def build_HR2(r1, r2, r3, t1, t2, H1, H2, o, v):
    """Compute the projection of HR on 3h-1p excitations
        X[i, j, c, k] = < ijkc | [ HBar(CCSD) * (R1 + R2 + R3) ]_C | 0 >
    """
    I_vo = (
            0.5 * np.einsum("mnie,mn->ie", H2[o, o, o, v], r1, optimize=True)
            - 0.5 * np.einsum("mnef,jnem->jf", H2[o, o, v, v], r2, optimize=True)
    )

    X2 = (3.0 / 6.0) * np.einsum("ie,ecjk->ijck", I_vo, t2, optimize=True)
    X2 -= (3.0 / 6.0) * np.einsum("cmki,mj->ijck", H2[v, o, o, o], r1, optimize=True)
    X2 += (1.0 / 6.0) * np.einsum("ce,ijek->ijck", H1[v, v], r2, optimize=True)
    X2 -= (3.0 / 6.0) * np.einsum("mk,ijcm->ijck", H1[o, o], r2, optimize=True)
    X2 += (3.0 / 12.0) * np.einsum("mnij,mnck->ijck", H2[o, o, o, o], r2, optimize=True)
    X2 += (3.0 / 6.0) * np.einsum("cmke,ijem->ijck", H2[v, o, o, v], r2, optimize=True)
    # terms contracted with R(4p-2h)
    X2 += (1.0 / 6.0) * np.einsum("me,ijcekm->ijck", H1[o, v], r3, optimize=True)
    X2 -= (3.0 / 12.0) * np.einsum("mnkf,ijcfmn->ijck", H2[o, o, o, v], r3, optimize=True)
    X2 += (1.0 / 12.0) * np.einsum("cnef,ijefkn->ijck", H2[v, o, v, v], r3, optimize=True)
    # antisymmetrize A(ijk)
    X2 -= np.transpose(X2, (0, 3, 2, 1)) # A(jk)
    X2 -= np.transpose(X2, (1, 0, 2, 3)) + np.transpose(X2, (3, 1, 2, 0)) # A(i/jk)
    return X2

def build_HR3(r1, r2, r3, t1, t2, H1, H2, o, v):
    """Compute the projection of HR on 4h-2p excitations
        X[i, j, c, d, k, l] = < ijklcd | [ HBar(CCSD) * (R1 + R2 + R3) ]_C | 0 >
    """
    ### Explicit usage of 3-body Hbar ###
    # I(abcefk)
    #I_vvvvvo = (
    #    # A(a/bc) -h(anef) t2(bcnk)
    #    -np.einsum("anef,bcnk->abcefk", H2[v, o, v, v], t2, optimize=True)
    #)
    #I_vvvvvo -= np.transpose(I_vvvvvo, (1, 0, 2, 3, 4, 5)) + np.transpose(I_vvvvvo, (2, 1, 0, 3, 4, 5))
    # I(abmije)
    #I_vvooov = (
    #    # A(ab) h(bmfe) t2(afij)
    #    (1.0 / 2.0) * np.einsum("bmfe,afij->abmije", H2[v, o, v, v], t2, optimize=True)
    #    # -A(ij) h(nmje) t2(abin)
    #    - (1.0 / 2.0) * np.einsum("nmje,abin->abmije", H2[o, o, o, v], t2, optimize=True)
    #)
    #I_vvooov -= np.transpose(I_vvooov, (1, 0, 2, 3, 4, 5))
    #I_vvooov -= np.transpose(I_vvooov, (0, 1, 2, 4, 3, 5))
    # I(abcije)
    #I_vvvoov = (
    #    # -A(ij)A(c/ab) h(bmje) t2(acim)
    #    - (6.0 / 12.0) * np.einsum("bmje,acim->abcije", H2[v, o, o, v], t2, optimize=True)
    #    # A(a/bc) h(bcfe) t2(afij)
    #    + (3.0 / 12.0) * np.einsum("bcfe,afij->abcije", H2[v, v, v, v], t2, optimize=True)
    #)
    #I_vvvoov -= np.transpose(I_vvvoov, (0, 1, 2, 4, 3, 5)) # A(ij)
    #I_vvvoov -= np.transpose(I_vvvoov, (0, 2, 1, 3, 4, 5)) # A(bc)
    #I_vvvoov -= np.transpose(I_vvvoov, (1, 0, 2, 3, 4, 5)) + np.transpose(I_vvvoov, (2, 1, 0, 3, 4, 5)) # A(a/bc)
    # I(amnijf)
    #I_voooov = (
    #        # h(mnef) t2(aeij)
    #        np.einsum("mnef,aeij->amnijf", H2[o, o, v, v], t2, optimize=True)
    #)
    # I(abnief)
    #I_vvoovv = (
    #        # h(mnef) t2(abim)
    #        -np.einsum("mnef,abim->abnief", H2[o, o, v, v], t2, optimize=True)
    #)
    ####
    # Intermediates
    I_vv = (
            0.5 * np.einsum("mnef,mn->ef", H2[o, o, v, v], r1, optimize=True)
    )
    # I(ijmk)
    I_oooo = (
          (3.0 / 6.0) * np.einsum("nmke,ijem->ijnk", H2[o, o, o, v], r2, optimize=True)
        - (3.0 / 6.0) * np.einsum("mnik,mj->ijnk", H2[o, o, o, o], r1, optimize=True)
        + (1.0 / 12.0) * np.einsum("mnef,ijefkn->ijmk", H2[o, o, v, v], r3, optimize=True)
    )
    # antisymmetrize A(ijk)
    I_oooo -= np.transpose(I_oooo, (0, 3, 2, 1)) # A(jk)
    I_oooo -= np.transpose(I_oooo, (1, 0, 2, 3)) + np.transpose(I_oooo, (3, 1, 2, 0)) # A(i/jk)
    # I(ijce)
    I_oovv = (
        (1.0 / 2.0) * np.einsum("cmfe,ijem->ijcf", H2[v, o, v, v], r2, optimize=True)
        + np.einsum("bmje,mk->jkbe", H2[v, o, o, v], r1, optimize=True)
        + 0.5 * np.einsum("nmie,njcm->ijce", H2[o, o, o, v], r2, optimize=True)
        - (1.0 / 4.0) * np.einsum("mnef,ijcfmn->ijce", H2[o, o, v, v], r3, optimize=True)
        #+ np.einsum("ef,edil->ilfd", I_vv, t2, optimize=True) # should include 4-body Hbar here somehow
    )
    # antisymmetrize A(ij)
    I_oovv -= np.transpose(I_oovv, (1, 0, 2, 3))
    #
    # Moment-like terms
    X3 = (4.0 / 48.0) * np.einsum("dcle,ijek->ijcdkl", H2[v, v, o, v], r2, optimize=True)
    X3 -= (12.0 / 48.0) * np.einsum("dmlk,ijcm->ijcdkl", H2[v, o, o, o], r2, optimize=True)
    X3 -= (4.0 / 48.0) * np.einsum("ijmk,cdml->ijcdkl", I_oooo, t2, optimize=True)
    X3 += (12.0 / 48.0) * np.einsum("ijce,edkl->ijcdkl", I_oovv, t2, optimize=True)
    # 
    X3 += (2.0 / 48.0) * np.einsum("de,ijcekl->ijcdkl", H1[v, v], r3, optimize=True)
    X3 -= (4.0 / 48.0) * np.einsum("mi,mjcdkl->ijcdkl", H1[o, o], r3, optimize=True)
    X3 += (1.0 / 96.0) * np.einsum("cdef,ijefkl->ijcdkl", H2[v, v, v, v], r3, optimize=True)
    X3 += (6.0 / 96.0) * np.einsum("mnij,mncdkl->ijcdkl", H2[o, o, o, o], r3, optimize=True)
    X3 += (8.0 / 48.0) * np.einsum("dmle,ijcekm->ijcdkl", H2[v, o, o, v], r3, optimize=True)
    ### Explicit usage of 3-body Hbar ###
    #X3 += (6.0 / 48.0) * np.einsum("cdmkle,abem->abcdkl", I_vvooov, r2, optimize=True)
    #X3 += (8.0 / 96.0) * np.einsum("abcefk,efdl->abcdkl", I_vvvvvo, r2, optimize=True)
    #X3 += (4.0 / 48.0) * np.einsum("cdbkle,ae->abcdkl", I_vvvoov, r1, optimize=True)
    #X3 -= (4.0 / 96.0) * np.einsum("dmnlkf,abcfmn->abcdkl", I_voooov, r3, optimize=True)
    #X3 += (12.0 / 96.0) * np.einsum("dcnlef,abefkn->abcdkl", I_vvoovv, r3, optimize=True)
    ####
    ### 4-body HBar ###
    X3 += (6.0 / 48.0) * np.einsum("ef,edil,fcjk->ijcdkl", I_vv, t2, t2, optimize=True)
    # antisymmetrize A(ijkl)A(cd)
    X3 -= np.transpose(X3, (0, 1, 3, 2, 4, 5)) # A(cd)
    X3 -= np.transpose(X3, (0, 4, 2, 3, 1, 5)) # A(jk)
    X3 -= np.transpose(X3, (1, 0, 2, 3, 4, 5)) + np.transpose(X3, (4, 1, 2, 3, 0, 5)) # A(i/jk)
    X3 -= np.transpose(X3, (5, 1, 2, 3, 4, 0)) + np.transpose(X3, (0, 5, 2, 3, 4, 1)) + np.transpose(X3, (0, 1, 2, 3, 5, 4)) # A(l/ijk)
    return X3

