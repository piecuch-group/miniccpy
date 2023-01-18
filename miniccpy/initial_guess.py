import numpy as np

def get_initial_guess(f, g, o, v, nroot):
    """Obtain the lowest `nroot` roots of the CIS Hamiltonian
    to serve as the initial guesses for the EOMCC calculations."""

    H = build_cis_hamiltonian(f, g, o, v)
    omega, C = np.linalg.eig(H)
    idx = np.argsort(omega)
    omega = omega[idx]
    C = C[:, idx]

    # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
    R_guess, _ = np.linalg.qr(C[:, :nroot])

    return R_guess, omega[:nroot]


def build_cis_hamiltonian(f, g, o, v):
    """ Construct the CIS Hamiltonian with matrix elements
        given by:
        < ia | H_N | jb > = < a | f | b > * delta(i, j)
                          - < j | f | i > * delta(a, b)
                          + < aj | v | ib >
    """

    nunocc, nocc = f[v, o].shape
    n1 = nocc * nunocc

    H = np.zeros((n1, n1))

    ct1 = 0 
    for a in range(nunocc):
        for i in range(nocc):
            ct2 = 0
            for b in range(nunocc):
                for j in range(nocc):
                    H[ct1, ct2] = (
                          f[v, v][a, b] * (i == j)
                        - f[o, o][j, i] * (a == b)
                        + g[v, o, o, v][a, j, i, b]
                    )
                    ct2 += 1
            ct1 += 1

    return H

