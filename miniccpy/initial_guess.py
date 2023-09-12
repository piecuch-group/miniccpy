import numpy as np

def cis_guess(f, g, o, v, nroot, mult=-1):
    """Obtain the lowest `nroot` roots of the CIS Hamiltonian
    to serve as the initial guesses for the EOMCC calculations."""

    # Decide whether reference is a closed shell or not
    nu, no = f[v, o].shape
    if no % 2 == 0:
        is_closed_shell = True
    else:
        is_closed_shell = False

    H = build_cis_hamiltonian(f, g, o, v)
    omega, C = np.linalg.eig(H)
    idx = np.argsort(omega)
    omega = omega[idx]
    C = C[:, idx]

    # For closed shells, we can pick out singlets and triplets numerically
    if is_closed_shell and mult != -1:
        omega_guess = np.zeros(nroot)
        C_spin = np.zeros((C.shape[0], nroot))
        n_spin = 0
        for n in range(C.shape[1]):
            if spin_function(C[:, n], mult, no, nu) <= 1.0e-06:
                C_spin[:, n_spin] = C[:, n]
                omega_guess[n_spin] = omega[n]
                n_spin += 1
            if n_spin >= nroot:
                break
        # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
        R_guess, _ = np.linalg.qr(C_spin[:, :min(n_spin, nroot)])
        omega_guess = omega_guess[:min(n_spin, nroot)]
    else:
        # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
        R_guess, _ = np.linalg.qr(C[:, :nroot])
        omega_guess = omega[:nroot]

    return R_guess, omega_guess

def eacis_guess(f, g, o, v, nroot):
    """Obtain the lowest `nroot` roots of the 1p Hamiltonian
    to serve as the initial guesses for the EA-EOMCC calculations."""

    H = build_1p_hamiltonian(f, g, o, v)
    omega, C = np.linalg.eig(H)
    idx = np.argsort(omega)
    omega = omega[idx]
    C = C[:, idx]

    # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
    R_guess, _ = np.linalg.qr(C)

    return R_guess[:, :nroot], omega[:nroot]

def ipcis_guess(f, g, o, v, nroot):
    """Obtain the lowest `nroot` roots of the 1h Hamiltonian
    to serve as the initial guesses for the IP-EOMCC calculations."""

    H = build_1h_hamiltonian(f, g, o, v)
    omega, C = np.linalg.eig(H)
    idx = np.argsort(omega)
    omega = omega[idx]
    C = C[:, idx]

    # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
    R_guess, _ = np.linalg.qr(C)

    return R_guess[:, :nroot], omega[:nroot]

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

def build_1p_hamiltonian(f, g, o, v):
    """ Construct the CIS Hamiltonian with matrix elements
        given by:
        < a | H_N | b > = < a | f | b >
    """

    nunocc, nocc = f[v, o].shape

    H = np.zeros((nunocc, nunocc))

    ct1 = 0 
    for a in range(nunocc):
        ct2 = 0
        for b in range(nunocc):
            H[ct1, ct2] = f[v, v][a, b]
            ct2 += 1
        ct1 += 1

    return H

def build_1h_hamiltonian(f, g, o, v):
    """ Construct the CIS Hamiltonian with matrix elements
        given by:
        < i | H_N | j > = -< j | f | i >
    """

    nunocc, nocc = f[v, o].shape

    H = np.zeros((nocc, nocc))

    ct1 = 0 
    for i in range(nocc):
        ct2 = 0
        for j in range(nocc):
            H[ct1, ct2] = -f[o, o][j, i]
            ct2 += 1
        ct1 += 1

    return H

def spin_function(C1, mult, no, nu):
    # Reshape the excitation vector into C1
    c1_arr = np.reshape(np.real(C1), (nu, no))
    # Create the a->a and b->b single excitation cases
    c1_a = np.zeros((nu // 2, no // 2))
    c1_b = np.zeros((nu // 2, no // 2))
    for a in range(nu):
        for i in range(no):
            if a % 2 == 0 and i % 2 == 0:
                c1_a[a // 2, i // 2] = c1_arr[a, i]
            elif a % 2 == 1 and i % 2 == 1:
                c1_b[ (a - 1) // 2, (i - 1) // 2] = c1_arr[a, i]
    # For RHF, singlets have c1_a = c1_b and triplets have c1_a = -c1_b
    if mult == 1:
        error = np.linalg.norm(c1_a - c1_b)
    elif mult == 3:
        error = np.linalg.norm(c1_a + c1_b)
    return error
