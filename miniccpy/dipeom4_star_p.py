import time
import numpy as np
import h5py
from miniccpy.utilities import get_memory_usage, remove_file
from miniccpy.lib import dipeom4_star_p

# IMPORTANT NOTE:
# r3_excitations must be passed back from the HR function. Otherwise, it will not
# update and r3_amps/HR3 and r3_excitations will be out of alignment. This behavior
# can be checked using the tmp variables.

def kernel(R0, T, omega, fock, g, H1, H2, o, v, r3_excitations=None, maxit=80, convergence=1.0e-07, max_size=20, nrest=1, out_of_core=False):
    """
    Diagonalize the similarity-transformed CCSD Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
    from miniccpy.energy import calc_rel_dip

    remove_file("eomcc-vectors.hdf5")
    if out_of_core:
        f = h5py.File("eomcc-vectors.hdf5", "w")

    # determine whether r3 updates should be done
    do_r3 = True
    if np.array_equal(r3_excitations[0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_r3 = False

    t1, t2 = T

    nunocc, nocc = t1.shape
    n1 = nocc**2
    n2 = nocc**3 * nunocc
    n3 = r3_excitations.shape[0]
    ndim = n1 + n2 + n3
    
    if len(R0) <= ndim:
        R = np.zeros(ndim)
        R[:len(R0)] = R0

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
    sigma[0, :], r3_excitations = HR(R[:n1].reshape(nocc, nocc),
                     R[n1:n1+n2].reshape(nocc, nocc, nunocc, nocc),
                     R[n1+n2:], r3_excitations,
                     t1, t2, fock, g, H1, H2, o, v, do_r3)

    print("    ==> DIP-EOMCC(4h-2p)(P)* iterations <==")
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
        q = update(residual[:n1].reshape(nocc, nocc),
                   residual[n1:n1+n2].reshape(nocc, nocc, nunocc, nocc),
                   residual[n1+n2:], r3_excitations,
                   omega,
                   fock[o, o], fock[v, v],
                   H1[o, o], H1[v, v])
        for p in range(curr_size):
            b = B[p, :] / np.linalg.norm(B[p, :])
            q -= np.dot(b, q) * b
        q /= np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[curr_size, :] = q
            sigma[curr_size, :], r3_excitations = HR(q[:n1].reshape(nocc, nocc),
                                     q[n1:n1+n2].reshape(nocc, nocc, nunocc, nocc),
                                     q[n1+n2:], r3_excitations,
                                     t1, t2, fock, g, H1, H2, o, v, do_r3)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[j, :] = restart_block[:, j]
                sigma[j, :], r3_excitations = HR(restart_block[:n1, j].reshape(nocc, nocc),
                                 restart_block[n1:n1+n2, j].reshape(nocc, nocc, nunocc, nocc),
                                 restart_block[n1+n2:, j], r3_excitations,
                                 t1, t2, fock, g, H1, H2, o, v, do_r3)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        print("DIP-EOMCC(4h-2p)(P)* iterations did not converge")

    # Save the final converged root in an excitation tuple
    R = (R[:n1].reshape(nocc, nocc), R[n1:n1+n2].reshape(nocc, nocc, nunocc, nocc), R[n1+n2:])
    # r0 for a root in DIP is 0 by definition
    r0 = 0.0
    # Compute relative excitation level diagnostic
    rel = calc_rel_dip(R[0], R[1])
    # remove the HDF5 file
    remove_file("eomcc-vectors.hdf5")
    return R, omega, r0, rel

def update(r1, r2, r3, r3_excitations, omega, fock_oo, fock_vv, h1_oo, h1_vv):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""
    r1, r2, r3 = dipeom4_star_p.dipeom4_star_p.update_r(r1, r2, r3, r3_excitations, omega, fock_oo, fock_vv, h1_oo, h1_vv)
    return np.hstack([r1.flatten(), r2.flatten(), r3])

def HR(r1, r2, r3, r3_excitations, t1, t2, fock, g, H1, H2, o, v, do_r3):
    """Compute the matrix-vector product H * R, where
    H is the CCSD similarity-transformed Hamiltonian and R is
    the DIP-EOMCC linear excitation operator."""
    # update R1
    HR1 = build_HR1(r1, r2, r3, r3_excitations, H1, H2, o, v)
    # update R2
    HR2 = build_HR2(r1, r2, r3, r3_excitations, t1, t2, H1, H2, o, v)
    # update R3
    if do_r3:
        HR3, r3_excitations = build_HR3(r1, r2, r3, r3_excitations, t1, t2, fock, g, H1, H2, o, v)
    return np.hstack([HR1.flatten(), HR2.flatten(), HR3]), r3_excitations

def build_HR1(r1, r2, r3, r3_excitations, H1, H2, o, v):
    """Compute the projection of HR on 2h excitations
        X[i, j] = < ij | [ HBar(CCSD) * (R1 + R2 + R3) ]_C | 0 >
    """
    X1 = -np.einsum("mi,mj->ij", H1[o, o], r1, optimize=True)
    X1 += 0.25 * np.einsum("mnij,mn->ij", H2[o, o, o, o], r1, optimize=True)
    X1 += 0.5 * np.einsum("me,ijem->ij", H1[o, v], r2, optimize=True)
    X1 -= 0.5 * np.einsum("mnif,mjfn->ij", H2[o, o, o, v], r2, optimize=True)
    X1 = dipeom4_star_p.dipeom4_star_p.build_hr1(X1, r3, r3_excitations, H2[o, o, v, v])
    return X1

def build_HR2(r1, r2, r3, r3_excitations, t1, t2, H1, H2, o, v):
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
    X2 = dipeom4_star_p.dipeom4_star_p.build_hr2(X2, r3, r3_excitations, H1[o, v], H2[v, o, v, v], H2[o, o, o, v])
    return X2

def build_HR3(r1, r2, r3, r3_excitations, t1, t2, fock, g, H1, H2, o, v):
    """Compute the projection of HR on 4h-2p excitations
        X[i, j, c, d, k, l] = < ijklcd | [ HBar(CCSD) * (R1 + R2 + R3) ]_C | 0 >,
        approximated complete to 3rd-order in MBPT (assuming 2h is 0th order). The
        resulting terms include (H[2]*R1)_C + (H[1]*R2)_C + (F_N*R3)_C.
    """
    # Intermediates
    # This term factorizes the 4-body HBar formed by (V_N*T2^2)_C, which enters at 3rd-order.
    # This should be removed too
    #I_vv = (
    #        0.5 * np.einsum("mnef,mn->ef", H2[o, o, v, v], r1, optimize=True)
    #)

    # I(ijmk)
    I_oooo = (
          (3.0 / 6.0) * np.einsum("nmke,ijem->ijnk", H2[o, o, o, v], r2, optimize=True) # includes T1
        - (3.0 / 6.0) * np.einsum("mnik,mj->ijnk", H2[o, o, o, o], r1, optimize=True) # includes T1 and T2
    )
    # antisymmetrize A(ijk)
    I_oooo -= np.transpose(I_oooo, (0, 3, 2, 1)) # A(jk)
    I_oooo -= np.transpose(I_oooo, (1, 0, 2, 3)) + np.transpose(I_oooo, (3, 1, 2, 0)) # A(i/jk)
    # This (V_N*R3)_C term is removed
    #I_oooo = dipeom4_star_p.dipeom4_star_p.build_i_oooo(I_oooo, r3, r3_excitations, H2[o, o, v, v])

    # I(ijce)
    I_oovv = (
        (1.0 / 2.0) * np.einsum("cmfe,ijem->ijcf", H2[v, o, v, v], r2, optimize=True) # includes T1
        + np.einsum("bmje,mk->jkbe", H2[v, o, o, v], r1, optimize=True) # includes T1 and T2
        + 0.5 * np.einsum("nmie,njcm->ijce", H2[o, o, o, v], r2, optimize=True) # includes T1
        # + 0.25 * np.einsum("ef,edil->lidf", I_vv, t2, optimize=True) # remove 4-body HBar term
    )
    # antisymmetrize A(ij)
    I_oovv -= np.transpose(I_oovv, (1, 0, 2, 3))
    # This (V_N*R3)_C term is removed
    #I_oovv = dipeom4_star_p.dipeom4_star_p.build_i_oovv(I_oovv, r3, r3_excitations, H2[o, o, v, v])

    X3, r3, r3_excitations = dipeom4_star_p.dipeom4_star_p.build_hr4_p(
            r3, r3_excitations,
            t2, r2,
            fock[o, o], fock[v, v],
            g[v, v, o, v], g[v, o, o, o], I_oooo, I_oovv,
    )
    return X3, r3_excitations
