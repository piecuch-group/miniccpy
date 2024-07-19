import time
import numpy as np
import h5py
from miniccpy.utilities import get_memory_usage, remove_file

def kernel(R0, T, omega, fock, g, H1, H2, o, v, maxit=80, convergence=1.0e-07, max_size=20, nrest=1, out_of_core=False):
    """
    Diagonalize the similarity-transformed CCSDT Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
    from miniccpy.energy import calc_r0_rhf, calc_rel_rhf

    remove_file("eomcc-vectors.hdf5")
    if out_of_core:
        f = h5py.File("eomcc-vectors.hdf5", "w")

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
    if out_of_core:
        sigma = f.create_dataset("sigma", (max_size, ndim), dtype=np.float64)
        B = f.create_dataset("bmatrix", (max_size, ndim), dtype=np.float64)
    else:
        sigma = np.zeros((max_size, ndim))
        B = np.zeros((max_size, ndim))

    restart_block = np.zeros((ndim, nrest))
    G = np.zeros((max_size, max_size))

    # Initial values
    B[0, :] = R
    sigma[0, :] = HR(R[:n1].reshape(nunocc, nocc),
                     R[n1:n1+n2].reshape(nunocc, nunocc, nocc, nocc),
                     R[n1+n2:].reshape(nunocc, nunocc, nunocc, nocc, nocc, nocc),
                     t1, t2, t3, fock, g, H1, H2, o, v)

    print("    ==> R-EOMCCSDT(a) iterations <==")
    print("    The initial guess energy = ", omega)
    print("")
    print("     Iter               Energy                 |dE|                 |dR|     Wall Time     Memory")
    curr_size = 1
    for niter in range(maxit):
        tic = time.time()
        # store old energy
        omega_old = omega

        # solve projection subspace eigenproblem: G_{IJ} = sum_K B_{KI} S_{KJ} (vectorized)
        G[curr_size - 1, :curr_size] = np.einsum("k,pk->p", B[curr_size - 1, :], sigma[:curr_size, :])
        G[:curr_size, curr_size - 1] = np.einsum("k,pk->p", sigma[curr_size - 1, :], B[:curr_size, :])
        e, alpha_full = np.linalg.eig(G[:curr_size, :curr_size])

        # select root based on maximum overlap with initial guess
        idx = np.argsort(abs(alpha_full[0, :]))
        iselect = idx[-1]

        alpha = np.real(alpha_full[:, iselect])

        # Get the eigenpair of interest
        omega = np.real(e[iselect])
        R = np.dot(B[:curr_size, :].T, alpha)
        restart_block[:, niter % nrest] = R

        # calculate residual vector
        residual = np.dot(sigma[:curr_size, :].T, alpha) - omega * R
        res_norm = np.linalg.norm(residual)
        delta_e = omega - omega_old

        if res_norm < convergence and abs(delta_e) < convergence:
            toc = time.time()
            minutes, seconds = divmod(toc - tic, 60)
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
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
            b = B[p, :] / np.linalg.norm(B[p, :])
            q -= np.dot(b.T, q) * b
        q *= 1.0 / np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[curr_size, :] = q
            sigma[curr_size, :] = HR(q[:n1].reshape(nunocc, nocc),
                                     q[n1:n1+n2].reshape(nunocc, nunocc, nocc, nocc),
                                     q[n1+n2:].reshape(nunocc, nunocc, nunocc, nocc, nocc, nocc),
                                     t1, t2, t3, fock, g, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[j, :] = restart_block[:, j]
                sigma[j, :] = HR(restart_block[:n1, j].reshape(nunocc, nocc),
                                 restart_block[n1:n1+n2, j].reshape(nunocc, nunocc, nocc, nocc),
                                 restart_block[n1+n2:, j].reshape(nunocc, nunocc, nunocc, nocc, nocc, nocc),
                                 t1, t2, t3, fock, g, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        print("EOMCCSDT(a) iterations did not converge")

    # Save the final converged root in an excitation tuple
    R = (R[:n1].reshape(nunocc, nocc), R[n1:n1+n2].reshape(nunocc, nunocc, nocc, nocc), R[n1+n2:].reshape(nunocc, nunocc, nunocc, nocc, nocc, nocc))
    # Calculate r0 for the root
    r0 = calc_r0_rhf(R[0], R[1], H1, H2, omega, o, v)
    # Compute relative excitation level diagnostic
    rel = calc_rel_rhf(r0, R[0], R[1])
    # remove the HDF5 file
    remove_file("eomcc-vectors.hdf5")
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
    HR3 = build_HR3(r1, r2, r3, t1, t2, t3, fock, g, H1, H2, o, v)

    return np.hstack( [HR1.flatten(), HR2.flatten(), HR3.flatten()] )


def build_HR1(r1, r2, r3, H1, H2, o, v):
    """Compute the projection of HR on singles
        X[a, i] = < ia | [ HBar(CCSD(T)(a)) * (R1 + R2 + R3) ]_C | 0 >
    """
    # spin-summed t3: t3(ABcIJk) = 2*t3(AbcIjk) - t3(Abc
    #                            = 2*(2*t3(abcijk) - t3(abcjik) - t3(abckji)) - 
    r3_ss = (
                4.0 * r3 
                - 2.0 * r3.transpose(0, 1, 2, 4, 3, 5) 
                - 2.0 * r3.transpose(0, 1, 2, 5, 4, 3) 
                - 2.0 * r3.transpose(0, 1, 2, 3, 5, 4) 
                + r3.transpose(0, 1, 2, 4, 5, 3) 
                + r3.transpose(0, 1, 2, 5, 3, 4)
    )
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
    # R3 parts
    X1 += 0.5 * np.einsum("mnef,aefimn->ai", H2[o, o, v, v], r3_ss, optimize=True)
    return X1


def build_HR2(r1, r2, r3, t1, t2, t3, H1, H2, o, v):
    """Compute the projection of HR on doubles
        X[a, b, i, j] = < ijab | [ HBar(CCSD(T)(a)) * (R1 + R2 + R3) ]_C | 0 >
    """
    # partially spin-summed t3(AbcIjk) = 2*t3(abcijk) - t3(abcjik) - t3(abckji)
    t3_s = (
            2.0 * t3
            - t3.transpose(0, 1, 2, 4, 3, 5)
            - t3.transpose(0, 1, 2, 5, 4, 3)
    )
    r3_s = (
            2.0 * r3
            - r3.transpose(0, 1, 2, 4, 3, 5)
            - r3.transpose(0, 1, 2, 5, 4, 3)
    )
    # intermediates
    X_ov = (
          2.0 * np.einsum("mnef,fn->me", H2[o, o, v, v], r1, optimize=True)
        - np.einsum("nmef,fn->me", H2[o, o, v, v], r1, optimize=True)
    )
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
    # T3 parts
    X2 += 0.5 * np.einsum("me,eabmij->abij", X_ov, t3_s, optimize=True)
    # R3 parts
    X2 += 0.5 * np.einsum("me,eabmij->abij", H1[o, v], r3_s, optimize=True)
    X2 += np.einsum("amef,febmij->abij", H2[v, o, v, v], r3_s, optimize=True)
    X2 -= np.einsum("nmje,eabmin->abij", H2[o, o, o, v], r3_s, optimize=True)
    X2 += X2.transpose(1, 0, 3, 2)
    return X2

def build_HR3(r1, r2, r3, t1, t2, t3, fock, g, H1, H2, o, v):
    """Compute the projection of HR on triples
        X[a, b, c, i, j, k] = < ijkabc | [ HBar(CCSD(T)(a)) * (R1 + R2 + R3) ]_C | 0 >
        H_TS = H_TS(corr) + [F_N + V_N, T_3(MBPT)]
        H_TD = H_TD(corr) + [V_N, T_3(MBPT)]

        H_DS = H_DS(corr) + [V_N, T_3(MBPT)] <- h(vvov) & h(vooo) terms; already taken care of in HBar CCSD(T)(a)
    """

    # partially spin-summed t3(AbcIjk) = 2*t3(abcijk) - t3(abcjik) - t3(abckji)
    t3_s = (
            2.0 * t3
            - t3.transpose(0, 1, 2, 4, 3, 5)
            - t3.transpose(0, 1, 2, 5, 4, 3)
    )
    r3_s = (
            2.0 * r3
            - r3.transpose(0, 1, 2, 4, 3, 5)
            - r3.transpose(0, 1, 2, 5, 4, 3)
    )
    # Intermediates
    X1 = np.zeros_like(H1)
    X2 = np.zeros_like(H2)
    # From the above formulas that show the T_3(MBPT) modifications to H_TS, H_DT, and H_TD,
    # all expressions that involve T_3(MBPT) will use the bare Hamiltonian only!
    X1[o, v] = (
          2.0 * np.einsum("mnef,fn->me", g[o, o, v, v], r1, optimize=True)
        - np.einsum("nmef,fn->me", g[o, o, v, v], r1, optimize=True)
    )
    X1[o, o] = (
            + 2.0 * np.einsum("mnjf,fn->mj", g[o, o, o, v], r1, optimize=True)
            - np.einsum("nmjf,fn->mj", g[o, o, o, v], r1, optimize=True)
            + 2.0 * np.einsum("mnef,efjn->mj", g[o, o, v, v], r2, optimize=True)
            - np.einsum("nmef,efjn->mj", g[o, o, v, v], r2, optimize=True)
            + np.einsum("me,ej->mj", fock[o, v], r1, optimize=True)
    )
    X1[v, v] = (
            + 2.0 * np.einsum("bnef,fn->be", g[v, o, v, v], r1, optimize=True)
            - np.einsum("bnfe,fn->be", g[v, o, v, v], r1, optimize=True)
            - 2.0 * np.einsum("mnef,bfmn->be", g[o, o, v, v], r2, optimize=True)
            + np.einsum("nmef,bfmn->be", g[o, o, v, v], r2, optimize=True)
            - np.einsum("me,bm->be", fock[o, v], r1, optimize=True)
    )
    X2[o, o, o, o] = (
        np.einsum("nmje,ek->nmjk", g[o, o, o, v], r1, optimize=True)
        + np.einsum("mnke,ej->nmjk", g[o, o, o, v], r1, optimize=True)
        + np.einsum("mnef,efjk->mnjk", g[o, o, v, v], r2, optimize=True)
    )
    X2[v, v, v, v] = (
        - np.einsum("bmfe,cm->bcfe", g[v, o, v, v], r1, optimize=True)
        - np.einsum("cmef,bm->bcfe", g[v, o, v, v], r1, optimize=True)
        + np.einsum("mnef,bcmn->bcef", g[o, o, v, v], r2, optimize=True)
    )
    X2[v, o, o, v] = (
        -1.0 * np.einsum("nmje,bn->bmje", g[o, o, o, v], r1, optimize=True)
        + np.einsum("bmfe,fj->bmje", g[v, o, v, v], r1, optimize=True)
        + np.einsum("nmfe,fcnk->cmke", g[o, o, v, v], r2, optimize=True)
        - np.einsum("nmfe,fckn->cmke", g[o, o, v, v], r2, optimize=True)
        + np.einsum("mnef,cfkn->cmke", g[o, o, v, v], r2, optimize=True)
        - np.einsum("nmef,cfkn->cmke", g[o, o, v, v], r2, optimize=True)
    )
    X2[v, o, v, o] = (
        np.einsum("bmfe,ek->bmfk", g[v, o, v, v], r1, optimize=True)
        - np.einsum("mnkf,bn->bmfk", g[o, o, o, v], r1, optimize=True)
        - np.einsum("mnef,bfmk->bnek", g[o, o, v, v], r2, optimize=True)
    )
    X2[v, v, o, v] = (
        - np.einsum("cmej,bm->bcje", g[v, o, v, o], r1, optimize=True)
        - np.einsum("bmje,cm->bcje", g[v, o, o, v], r1, optimize=True)
        + np.einsum("bcef,ej->bcjf", g[v, v, v, v], r1, optimize=True)
        + np.einsum("mnjf,bcmn->bcjf", g[o, o, o, v], r2, optimize=True)
        + np.einsum("cmfe,bejm->bcjf", g[v, o, v, v], r2, optimize=True)
        - np.einsum("cmfe,bemj->bcjf", g[v, o, v, v], r2, optimize=True)
        + np.einsum("cmfe,bejm->bcjf", g[v, o, v, v], r2, optimize=True)
        - np.einsum("cmef,bejm->bcjf", g[v, o, v, v], r2, optimize=True)
        - np.einsum("bmef,ecjm->bcjf", g[v, o, v, v], r2, optimize=True)
        - np.einsum("nmfe,fabnim->abie", g[o, o, v, v], r3_s, optimize=True)
        - np.einsum("me,bcjm->bcje", X1[o, v], t2, optimize=True) # counterterm, similar to CR-CC(2,3)
    )
    X2[v, o, o, o] = (
        - np.einsum("mnjk,bm->bnjk", g[o, o, o, o], r1, optimize=True)
        + np.einsum("bmje,ek->bmjk", g[v, o, o, v], r1, optimize=True)
        + np.einsum("bmek,ej->bmjk", g[v, o, v, o], r1, optimize=True)
        + np.einsum("bnef,efjk->bnjk", g[v, o, v, v], r2, optimize=True)
        + np.einsum("nmke,bejm->bnjk", g[o, o, o, v], r2, optimize=True)
        - np.einsum("nmke,bemj->bnjk", g[o, o, o, v], r2, optimize=True)
        + np.einsum("nmke,bejm->bnjk", g[o, o, o, v], r2, optimize=True)
        - np.einsum("mnke,bejm->bnjk", g[o, o, o, v], r2, optimize=True)
        - np.einsum("nmje,benk->bmjk", g[o, o, o, v], r2, optimize=True)
        + np.einsum("nmfe,faenij->amij", g[o, o, v, v], r3_s, optimize=True)
    )
    # MM(2,3)B
    # H_TS * R1 <- [[F_N + V_N, T_3(MBPT)], R1]
    # H_TD * R2 <- [[V_N, T_3(MBPT)], R2]
    X3 = -np.einsum("amij,bcmk->abcijk", X2[v, o, o, o], t2, optimize=True)
    X3 += np.einsum("abie,ecjk->abcijk", X2[v, v, o, v], t2, optimize=True)
    # H_TD * R2 <- (H_TS(corr) * R2)_C
    X3 -= np.einsum("amij,bcmk->abcijk", H2[v, o, o, o], r2, optimize=True)
    X3 += np.einsum("abie,ecjk->abcijk", H2[v, v, o, v], r2, optimize=True)
    # (HBar*R3)_C
    X3 += 0.5 * np.einsum("ae,ebcijk->abcijk", H1[v, v], r3, optimize=True)
    X3 -= 0.5 * np.einsum("mi,abcmjk->abcijk", H1[o, o], r3, optimize=True)
    X3 += 0.5 * np.einsum("mnij,abcmnk->abcijk", H2[o, o, o, o], r3, optimize=True)
    X3 += 0.5 * np.einsum("abef,efcijk->abcijk", H2[v, v, v, v], r3, optimize=True)
    X3 += 0.5 * np.einsum("amie,ebcmjk->abcijk", H2[v, o, o, v], r3_s, optimize=True)
    X3 -= 0.25 * np.einsum("amei,ebcmjk->abcijk", H2[v, o, v, o], r3_s, optimize=True)
    X3 -= 0.5 * np.einsum("amei,ebcjmk->abcijk", H2[v, o, v, o], r3, optimize=True)
    X3 -= np.einsum("bmei,eacjmk->abcijk", H2[v, o, v, o], r3, optimize=True)
    # (XBar*T3)_C
    X3 += 0.5 * np.einsum("ae,ebcijk->abcijk", X1[v, v], t3, optimize=True)
    X3 -= 0.5 * np.einsum("mi,abcmjk->abcijk", X1[o, o], t3, optimize=True)
    X3 += 0.5 * np.einsum("mnij,abcmnk->abcijk", X2[o, o, o, o], t3, optimize=True)
    X3 += 0.5 * np.einsum("abef,efcijk->abcijk", X2[v, v, v, v], t3, optimize=True)
    X3 += 0.5 * np.einsum("amie,ebcmjk->abcijk", X2[v, o, o, v], t3_s, optimize=True)
    X3 -= 0.25 * np.einsum("amei,ebcmjk->abcijk", X2[v, o, v, o], t3_s, optimize=True)
    X3 -= 0.5 * np.einsum("amei,ebcjmk->abcijk", X2[v, o, v, o], t3, optimize=True)
    X3 -= np.einsum("bmei,eacjmk->abcijk", X2[v, o, v, o], t3, optimize=True)

    # [1 + P(ai/bj)][1 + P(ai/ck) + P(bj/ck)] = 1 + P(ai/bj) + P(ai/ck) + P(bj/ck) + P(ai/bj)P(ai/ck) + P(ai/bj)P(bj/ck)
    X3 += (     X3.transpose(1, 0, 2, 4, 3, 5)   # (ij)(ab)
              + X3.transpose(2, 1, 0, 5, 4, 3)   # (ac)(ik)
              + X3.transpose(0, 2, 1, 3, 5, 4)   # (bc)(jk)
              + X3.transpose(2, 0, 1, 5, 3, 4)   # (ab)(ij)(ac)(ik)
              + X3.transpose(1, 2, 0, 4, 5, 3) ) # (ab)(ij)(bc)(jk)
    # Manually zero out the i = j = k and a = b = c blocks
    nu, no = r1.shape
    for i in range(no):
        X3[:, :, :, i, i, i] *= 0.0
    for a in range(nu):
        X3[a, a, a, :, :, :] *= 0.0
    return X3

