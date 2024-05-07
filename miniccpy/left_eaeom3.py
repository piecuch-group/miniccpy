import time
import numpy as np
from miniccpy.utilities import get_memory_usage

def build_LH1(l1, l2, l3, H1, H2, X, o, v):
    """Compute the projection of the CCSD Hamiltonian on 1p excitations
        X[a] = < 0 | (L1 + L2 + L3)*(H_N exp(T1+T2))_C | a >
    """
    LH = np.einsum("e,ea->a", l1, H1[v, v], optimize=True)
    LH += 0.5 * np.einsum("efn,fena->a", l2, H2[v, v, o, v], optimize=True)
    #
    # 3-body hbar
    # 
    LH -= np.einsum("fmna,mfn->a", H2[v, o, o, v], X["ovo"], optimize=True)
    LH -= 0.5 * np.einsum("fge,feag->a", X["vvv"], H2[v, v, v, v], optimize=True)
    # These are not the problem - I checked
    ## h3_vvvvoo = (
    ##     -(6.0 / 12.0) * np.einsum("bmje,acmk->abcejk", H2[v, o, o, v], t2, optimize=True)
    ##     +(3.0 / 12.0) * np.einsum("abef,fcjk->abcejk", H2[v, v, v, v], t2, optimize=True)
    ## )
    ## h3_vvvvoo -= np.transpose(h3_vvvvoo, (1, 0, 2, 3, 4, 5)) + np.transpose(h3_vvvvoo, (2, 1, 0, 3, 4, 5)) # antisymmetrize A(a/bc)
    ## h3_vvvvoo -= np.transpose(h3_vvvvoo, (0, 2, 1, 3, 4, 5)) # antisymmetrize A(bc)
    ## h3_vvvvoo -= np.transpose(h3_vvvvoo, (0, 1, 2, 3, 5, 4)) # antisymmetrize A(jk)
    ## LH += (1.0 / 12.0) * np.einsum("efgmn,efgamn->a", l3, h3_vvvvoo, optimize=True)
    #
    return LH


def build_LH2(l1, l2, l3, t2, H1, H2, X, o, v):
    """Compute the projection of the CCSD Hamiltonian on 2p1h excitations
        X[a, b, j] = < 0 | (L1 + L2 + L3)*(H_N exp(T1+T2))_C | abj >
    """
    LH = np.einsum("a,jb->abj", l1, H1[o, v], optimize=True)
    LH += 0.5 * np.einsum("e,ejab->abj", l1, H2[v, o, v, v], optimize=True)
    LH += np.einsum("ebj,ea->abj", l2, H1[v, v], optimize=True)
    LH -= 0.5 * np.einsum("abm,jm->abj", l2, H1[o, o], optimize=True)
    LH += np.einsum("afn,fjnb->abj", l2, H2[v, o, o, v], optimize=True)
    LH += 0.25 * np.einsum("efj,efab->abj", l2, H2[v, v, v, v], optimize=True)
    LH -= 0.5 * np.einsum("mjab,m->abj", H2[o, o, v, v], X["o"], optimize=True)
    #
    # parts with L3
    #
    LH += 0.5 * np.einsum("ebfjn,fena->abj", l3, H2[v, v, o, v], optimize=True)
    LH -= (1.0 / 4.0) * np.einsum("abfmn,fjnm->abj", l3, H2[v, o, o, o], optimize=True)
    #
    # 3-body hbar
    # 
    LH -= np.einsum("ejfb,afe->abj", H2[v, o, v, v], X["vvv"], optimize=True)
    LH += np.einsum("jmna,mbn->abj", H2[o, o, o, v], X["ovo"], optimize=True)
    LH -= 0.5 * np.einsum("emba,mej->abj", H2[v, o, v, v], X["ovo"], optimize=True)

    #h_vvooov = (
    #             -(2.0 / 4.0) * np.einsum("nmje,abin->abmije", H2[o, o, o, v], t2, optimize=True)
    #             +(2.0 / 4.0) * np.einsum("bmfe,afij->abmije", H2[v, o, v, v], t2, optimize=True)
    #)
    #h_vvooov -= np.transpose(h_vvooov, (1, 0, 2, 3, 4, 5)) # (ab)
    #h_vvooov -= np.transpose(h_vvooov, (0, 1, 2, 4, 3, 5)) # (ij)
    #LH += 0.25 * np.einsum("aefmn,efjmnb->abj", l3, h_vvooov, optimize=True)
    #
    #h_vvvvvo = -(3.0 / 6.0) * np.einsum("anef,bcnk->abcefk", H2[v, o, v, v], t2, optimize=True)
    #h_vvvvvo -= np.transpose(h_vvvvvo, (1, 0, 2, 3, 4, 5)) + np.transpose(h_vvvvvo, (2, 1, 0, 3, 4, 5)) # antisymmetrize A(a/bc)
    #h_vvvvvo -= np.transpose(h_vvvvvo, (0, 2, 1, 3, 4, 5)) # antisymmetrize A(bc)
    #LH += (1.0 / 12.0) * np.einsum("efgjo,efgabo->abj", l3, h_vvvvvo, optimize=True)
    #
    LH -= np.transpose(LH, (1, 0, 2))
    return LH

def build_LH3(l1, l2, l3, H1, H2, X, o, v):
    """Compute the projection of the CCSD Hamiltonian on 3p2h
        X[a, b, c, j, k] = < 0 | (L1 + L2 + L3)*(H_N exp(T1+T2))_C | abcjk >
    """
    # moment-like terms < 0 | (L1p+L2p1h)*(H_N e^(T1+T2))_C | jkabc >
    LH = (3.0 / 12.0) * np.einsum("a,jkbc->abcjk", l1, H2[o, o, v, v], optimize=True)
    LH += (6.0 / 12.0) * np.einsum("abj,kc->abcjk", l2, H1[o, v], optimize=True)
    LH -= (3.0 / 12.0) * np.einsum("abm,jkmc->abcjk", l2, H2[o, o, o, v], optimize=True)
    LH += (6.0 / 12.0) * np.einsum("eck,ejab->abcjk", l2, H2[v, o, v, v], optimize=True)
    # <0|L3p2h*(H_N e^(T1+T2))_C | jkabc>
    LH -= (2.0 / 12.0) * np.einsum("jm,abcmk->abcjk", H1[o, o], l3, optimize=True)
    LH += (3.0 / 12.0) * np.einsum("eb,aecjk->abcjk", H1[v, v], l3, optimize=True)
    LH += (1.0 / 24.0) * np.einsum("jkmn,abcmn->abcjk", H2[o, o, o, o], l3, optimize=True)
    LH += (3.0 / 24.0) * np.einsum("efbc,aefjk->abcjk", H2[v, v, v, v], l3, optimize=True)
    LH += (6.0 / 12.0) * np.einsum("ejmb,acekm->abcjk", H2[v, o, o, v], l3, optimize=True)
    #
    # 3-body hbar
    # 
    LH -= (6.0 / 12.0) * np.einsum("mck,mjab->abcjk", X["ovo"], H2[o, o, v, v], optimize=True)
    LH += (3.0 / 12.0) * np.einsum("aeb,jkec->abcjk", X["vvv"], H2[o, o, v, v], optimize=True)
    # These are not the problem - I checked
    ##
    ## h_voooov = np.einsum("mnef,aeij->amnijf", H2[o, o, v, v], t2, optimize=True)
    ## LH -= (3.0 / 24.0) * np.einsum("fjknmc,abfmn->abcjk", h_voooov, l3, optimize=True)
    ##
    ## h_vvoovv = -np.einsum("mnef,abim->abnief", H2[o, o, v, v], t2, optimize=True)
    ## LH += (6.0 / 24.0) * np.einsum("feknbc,aefjn->abcjk", h_vvoovv, l3, optimize=True)
    ##
    LH -= np.transpose(LH, (1, 0, 2, 3, 4)) + np.transpose(LH, (2, 1, 0, 3, 4)) # antisymmetrize A(a/bc)
    LH -= np.transpose(LH, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    LH -= np.transpose(LH, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return LH

def update(l1, l2, l3, omega, e_a, e_abj, e_abcjk):
    """Perform the diagonally preconditioned residual (DPR) update
    to get the next correction vector."""

    for a, d_a in enumerate(e_a):
        denom = omega - d_a
        if denom == 0: continue
        l1[a] /= denom
    #l1 /= (omega - e_a)
    l2 /= (omega - e_abj)
    l3 /= (omega - e_abcjk)

    return np.hstack([l1.flatten(), l2.flatten(), l3.flatten()])

def LH(l1, l2, l3, t1, t2, H1, H2, o, v):
    """Compute the matrix-vector product H * R, where
    H is the CCSD similarity-transformed Hamiltonian and R is
    the EOMCCSD linear excitation operator."""

    X = {"o": 0, "ovo": 0, "vvv": 0}
    # x1(i)
    X["o"] = 0.5 * np.einsum("efn,efmn->m", l2, t2, optimize=True)
    # x2(i|bj)
    X["ovo"] = 0.5 * np.einsum("ebfjn,efin->ibj", l3, t2, optimize=True)
    # x2(a|be)
    X["vvv"] = -0.5 * np.einsum("aefmn,bfmn->abe", l3, t2, optimize=True)

    LH1 = build_LH1(l1, l2, l3, H1, H2, X, o, v)
    LH2 = build_LH2(l1, l2, l3, t2, H1, H2, X, o, v)
    LH3 = build_LH3(l1, l2, l3, H1, H2, X, o, v)

    return np.hstack( [LH1.flatten(), LH2.flatten(), LH3.flatten()] )

def kernel(R, T, omega, H1, H2, o, v, maxit=80, convergence=1.0e-07, max_size=20, nrest=1):
    """
    Diagonalize the similarity-transformed CCSD Hamiltonian using the
    non-Hermitian Davidson algorithm for a specific root defined by an initial
    guess vector.
    """
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

    # Set the initial vector to be R
    r1, r2, r3 = R
    Rvec = np.hstack([r1.flatten(), r2.flatten(), r3.flatten()])
    L = Rvec.copy()

    # Allocate the B and sigma matrices
    sigma = np.zeros((ndim, max_size))
    B = np.zeros((ndim, max_size))
    restart_block = np.zeros((ndim, nrest))

    # Initial values
    B[:, 0] = L
    sigma[:, 0] = LH(L[:n1].reshape(nunocc),
                     L[n1:n1+n2].reshape(nunocc, nunocc, nocc),
                     L[n1+n2:].reshape(nunocc, nunocc, nunocc, nocc, nocc),
                     t1, t2, H1, H2, o, v)

    print("    ==> Left-EAEOMCC(3p-2h) iterations <==")
    print("    The initial guess energy = ", omega)
    print("")
    print("     Iter               Energy                 |dE|                 |dL|     Wall Time     Memory")
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
        L = np.dot(B[:, :curr_size], alpha)
        restart_block[:, niter % nrest] = L

        # calculate residual vector
        residual = np.dot(sigma[:, :curr_size], alpha) - omega * L
        res_norm = np.linalg.norm(residual)
        delta_e = omega - omega_old

        if res_norm < convergence and abs(delta_e) < convergence:
            toc = time.time()
            minutes, seconds = divmod(toc - tic, 60)
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
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
        q /= np.linalg.norm(q)

        # If below maximum subspace size, expand the subspace
        if curr_size < max_size:
            B[:, curr_size] = q
            sigma[:, curr_size] = LH(q[:n1].reshape(nunocc),
                                     q[n1:n1+n2].reshape(nunocc, nunocc, nocc),
                                     q[n1+n2:].reshape(nunocc, nunocc, nunocc, nocc, nocc),
                                     t1, t2, H1, H2, o, v)
        else:
            # Basic restart - use the last approximation to the eigenvector
            print("       **Deflating subspace**")
            restart_block, _ = np.linalg.qr(restart_block)
            for j in range(restart_block.shape[1]):
                B[:, j] = restart_block[:, j]
                sigma[:, j] = LH(restart_block[:n1, j].reshape(nunocc),
                                 restart_block[n1:n1+n2, j].reshape(nunocc, nunocc, nocc),
                                 restart_block[n1+n2:, j].reshape(nunocc, nunocc, nunocc, nocc, nocc),
                                 t1, t2, H1, H2, o, v)
            curr_size = restart_block.shape[1] - 1

        curr_size += 1

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s    {:.2f} MB".format(niter, omega, delta_e, res_norm, minutes, seconds, get_memory_usage()))
    else:
        print("Left-EAEOMCC(3p-2h) iterations did not converge")

    # Save the final converged root in an excitation tuple
    L = (L[:n1].reshape(nunocc), L[n1:n1+n2].reshape(nunocc, nunocc, nocc), L[n1+n2:].reshape(nunocc, nunocc, nunocc, nocc, nocc))

    return L, omega
