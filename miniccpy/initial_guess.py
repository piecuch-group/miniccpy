import numpy as np

def cisd_guess(f, g, o, v, nroot, nacto, nactu, mult=-1):
    """Obtain the lowest `nroot` roots of the CISd Hamiltonian
    to serve as the initial guesses for the EOMCC calculations."""

    nu, no = f[v, o].shape
    nacto = min(nacto, no)
    nactu = min(nactu, nu)
    # print dimensions of initial guess procedure
    print("   CISd initial guess")
    print("   Multiplicity = ", mult)
    print("   Number of roots = ", nroot)
    print("   Dimension of eigenvalue problem = ", no*nu + (nacto - 1)*nacto/2 * (nactu - 1)*nactu/2)
    print("   Active occupied = ", nacto)
    print("   Active unoccupied = ", nactu)
    print("   -----------------------------------")
    # Decide whether reference is a closed shell or not
    if no % 2 == 0:
        is_closed_shell = True
    else:
        is_closed_shell = False

    # Build the CISd Hamiltonian and diagonalize
    H = build_cisd_hamiltonian(f, g, o, v, nacto, nactu)
    omega, C_act = np.linalg.eig(H)
    idx = np.argsort(omega)
    omega = omega[idx]
    C_act = np.real(C_act[:, idx])

    # Scatter the active-space CISd vector into the full singles+doubles space
    nroot = min(nroot, C_act.shape[1])
    no, nu = f[o, v].shape
    ndim = no*nu + no**2*nu**2
    C = np.zeros((ndim, nroot))
    for i in range(nroot):
        C[:, i] = cisd_scatter(C_act[:, i], nacto, nactu, no, nu)

    # For closed shells, we can pick out singlets and triplets numerically
    if is_closed_shell and mult != -1:
        omega_guess = np.zeros(nroot)
        C_spin = np.zeros((C.shape[0], nroot))
        n_spin = 0
        for n in range(C.shape[1]):
            if spin_function(C[:no*nu, n], mult, no, nu) <= 1.0e-06:
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

    # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
    R_guess, _ = np.linalg.qr(C[:, :nroot])
    omega_guess = omega[:nroot]

    return R_guess, omega_guess

def cis_guess(f, g, o, v, nroot, mult=-1):
    """Obtain the lowest `nroot` roots of the CIS Hamiltonian
    to serve as the initial guesses for the EOMCC calculations."""

    nu, no = f[v, o].shape
    # print dimensions of initial guess procedure
    print("   CIS initial guess")
    print("   Multiplicity = ", mult)
    print("   Number of roots = ", nroot)
    print("   Dimension of eigenvalue problem = ", no*nu)
    print("   -----------------------------------")
    # Decide whether reference is a closed shell or not
    if no % 2 == 0:
        is_closed_shell = True
    else:
        is_closed_shell = False

    H = build_cis_hamiltonian(f, g, o, v)
    omega, C = np.linalg.eig(H)
    idx = np.argsort(omega)
    omega = omega[idx]
    C = np.real(C[:, idx])

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
    C = np.real(C[:, idx])

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
    C = np.real(C[:, idx])

    # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
    R_guess, _ = np.linalg.qr(C)

    return R_guess[:, :nroot], omega[:nroot]

def cisd_scatter(V_in, nacto, nactu, no, nu):

    # set active space parameters
    nacto = min(nacto, no)
    nactu = min(nactu, nu)

    # allocate full-length output vector
    V1_out = np.zeros((nu, no))
    V2_out = np.zeros((nu, nu, no, no))
    # Start filling in the array
    offset = 0
    for a in range(nu):
        for i in range(no):
            V1_out[a, i] = V_in[offset]
            offset += 1
    for a in range(nu):
        for b in range(a + 1, nu):
            for i in range(no):
                for j in range(i + 1, no):
                    if a < nactu and b < nactu and i >= no - nacto and j >= no - nacto:
                        V2_out[a, b, i, j] = V_in[offset]
                        V2_out[b, a, i, j] = -V_in[offset]
                        V2_out[a, b, j, i] = -V_in[offset]
                        V2_out[b, a, j, i] = V_in[offset]
                        offset += 1
    return np.hstack((V1_out.flatten(), V2_out.flatten()))

def build_cisd_hamiltonian(f, g, o, v, nacto, nactu):
    # Get orbital dimensions
    no, nu = f[o, v].shape
    # set active space parameters
    nacto = min(nacto, no)
    nactu = min(nactu, nu)
    # set dimensions of CISD problem
    n1 = no * nu
    n2 = int(nacto * (nacto - 1) / 2 * nactu * (nactu - 1) / 2)
    # total dimension
    ndim = n1 + n2
    
    # < ia | H | jb >
    s_H_s = np.zeros((n1, n1))
    ct1 = 0
    for a in range(nu):
        for i in range(no):
            ct2 = 0
            for b in range(nu):
                for j in range(no):
                    s_H_s[ct1, ct2] = (
                          f[v, v][a, b] * (i == j)
                        - f[o, o][j, i] * (a == b)
                        + g[v, o, o, v][a, j, i, b]
                    )
                    ct2 += 1
            ct1 += 1
    # < ia | H | jkbc >
    s_H_d = np.zeros((n1, n2))
    idet = 0
    for a in range(nu):
        for i in range(no):
            jdet = 0
            for b in range(nactu):
                for c in range(b + 1, nactu):
                    for j in range(no - nacto, no):
                        for k in range(j + 1, no):
                            hmatel = (
                               (i == k) * (a == c) * f[o, v][j, b]
                              +(i == j) * (a == b) * f[o, v][k, c]
                              -(i == j) * (a == c) * f[o, v][k, b]
                              -(i == k) * (a == b) * f[o, v][j, c]
                              -(a == b) * g[o, o, o, v][j, k, i, c]
                              +(a == c) * g[o, o, o, v][j, k, i, b]
                              +(i == j) * g[v, o, v, v][a, k, b, c]
                              -(i == k) * g[v, o, v, v][a, j, b, c]
                            )
                            s_H_d[idet, jdet] = hmatel
                            jdet += 1
            idet += 1
    # < ijab | H | kc >
    d_H_s = np.zeros((n2, n1))
    idet = 0
    for a in range(nactu):
        for b in range(a + 1, nactu):
            for i in range(no - nacto, no):
                for j in range(i + 1, no):
                    jdet = 0
                    for c in range(nu):
                        for k in range(no):
                            hmatel = (
                                 (j == k) * g[v, v, o, v][a, b, i, c]
                                +(i == k) * g[v, v, o, v][b, a, j, c]
                                -(b == c) * g[v, o, o, o][a, k, i, j]
                                -(a == c) * g[v, o, o, o][b, k, j, i]
                            )
                            d_H_s[idet, jdet] = hmatel
                            jdet += 1
                    idet += 1
    # < ijab | H | klcd >
    d_H_d = np.zeros((n2, n2))
    idet = 0
    for a in range(nactu):
        for b in range(a + 1, nactu):
            for i in range(no - nacto, no):
                for j in range(i + 1, no):
                    jdet = 0
                    for c in range(nactu):
                        for d in range(c + 1, nactu):
                            for k in range(no - nacto, no):
                                for l in range(k + 1, no):
                                    hmatel = (
                                        (a == c) * (b ==d) *
                                        (
                                                -(j == l) * f[o, o][k, i]
                                                +(i == l) * f[o, o][k, j]
                                                +(j == k) * f[o, o][l, i]
                                                -(i == k) * f[o, o][l, j]
                                        )
                                        + (j == l) * (i == k) *
                                        (
                                                + (b == d) * f[v, v][a, c]
                                                - (b == c) * f[v, v][a, d]
                                                + (a == c) * f[v, v][b, d]
                                                - (a == d) * f[v, v][b, c]
                                        )
                                        + (i == k) * (a == c) * g[v, o, o, v][b, l, j, d]
                                        - (i == k) * (a == d) * g[v, o, o, v][b, l, j, c]
                                        - (i == k) * (b == c) * g[v, o, o, v][a, l, j, d]
                                        + (i == k) * (b == d) * g[v, o, o, v][a, l, j, c]
                                        - (i == l) * (a == c) * g[v, o, o, v][b, k, j, d]
                                        + (i == l) * (a == d) * g[v, o, o, v][b, k, j, c]
                                        + (i == l) * (b == c) * g[v, o, o, v][a, k, j, d]
                                        - (i == l) * (b == d) * g[v, o, o, v][a, k, j, c]
                                        - (j == k) * (a == c) * g[v, o, o, v][b, l, i, d]
                                        + (j == k) * (a == d) * g[v, o, o, v][b, l, i, c]
                                        + (j == k) * (b == c) * g[v, o, o, v][a, l, i, d]
                                        - (j == k) * (b == d) * g[v, o, o, v][a, l, i, c]
                                        + (j == l) * (a == c) * g[v, o, o, v][b, k, i, d]
                                        - (j == l) * (a == d) * g[v, o, o, v][b, k, i, c]
                                        - (j == l) * (b == c) * g[v, o, o, v][a, k, i, d]
                                        + (j == l) * (b == d) * g[v, o, o, v][a, k, i, c]
                                        + (b == d) * (a == c) * g[o, o, o, o][k, l, i, j]
                                        + (i == k) * (j == l) * g[v, v, v, v][a, b, c, d]
                                    )
                                    d_H_d[idet, jdet] = hmatel
                                    jdet += 1
                    idet += 1
    # Assemble and return full matrix
    return np.concatenate(
        (np.concatenate((s_H_s, s_H_d), axis=1),
         np.concatenate((d_H_s, d_H_d), axis=1),
         ), axis=0
    )

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
