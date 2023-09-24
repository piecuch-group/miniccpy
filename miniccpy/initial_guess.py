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

    nroot = min(nroot, C_act.shape[1])
    no, nu = f[o, v].shape
    ndim = no*nu + no**2*nu**2
    C = np.zeros((ndim, nroot))
    # Scatter the active-space CISd vector into the full singles+doubles space
    if not is_closed_shell or mult != -1:
        for i in range(nroot):
            C[:, i] = cisd_scatter(C_act[:, i], nacto, nactu, no, nu)
        # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
        R_guess, _ = np.linalg.qr(C[:, :nroot])
        omega_guess = omega[:nroot]

    # For closed shells, we can pick out singlets and triplets numerically
    else:
        omega_guess = np.zeros(nroot)
        C_spin = np.zeros((C.shape[0], nroot))
        n_spin = 0
        for n in range(C.shape[1]):
            if spin_function(C_act[:no*nu, n], mult, no, nu) <= 1.0e-06:
                C_spin[:, n_spin] = cisd_scatter(C_act[:, n], nacto, nactu, no, nu)
                omega_guess[n_spin] = omega[n]
                n_spin += 1
            if n_spin >= nroot:
                break

        # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
        R_guess, _ = np.linalg.qr(C_spin[:, :min(n_spin, nroot)])
        omega_guess = omega_guess[:min(n_spin, nroot)]

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

def deacis_guess(f, g, o, v, nroot, nactu):
    """Obtain the lowest `nroot` roots of the 2p Hamiltonian
    to serve as the initial guesses for the DEA-EOMCC calculations."""

    H = build_2p_hamiltonian(f, g, o, v, nactu)
    omega, C = np.linalg.eig(H)
    idx = np.argsort(omega)
    omega = omega[idx]
    C = np.real(C[:, idx])

    no, nu = f[o, v].shape
    n1 = nu**2

    R_guess = np.zeros((n1, nroot))
    for i in range(nroot):
        R_guess[:, i] = deacis_scatter(C[:, i], nactu, no, nu)

    # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
    R_guess, _ = np.linalg.qr(R_guess)

    return R_guess, omega[:nroot]

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

def deacis_scatter(V_in, nactu, no, nu):

    # set active space parameters
    nactu = min(nactu, nu)
    # allocate full-length output vector
    V1_out = np.zeros((nu, nu))
    # Start filling in the array
    offset = 0
    for a in range(nu):
        for b in range(a + 1, nu):
            if a < nactu and b < nactu:
                V1_out[a, b] = V_in[offset]
                V1_out[b, a] = -V_in[offset]
                offset += 1
    return V1_out.flatten()


def build_cisd_hamiltonian(fock, g, o, v, nacto, nactu):

    no, nu = fock[o, v].shape

    # set dimensions of CISD problem
    n1 = no * nu
    n2 = int(nacto * (nacto - 1) / 2 * nactu * (nactu - 1) / 2)
    # total dimension
    ndim = n1 + n2
    # get index addressing arrays
    idx_1, idx_2 = get_index_arrays(no, nu, nacto, nactu)

    ###########
    # SINGLES #
    ###########
    s_H_s = np.zeros((n1, n1))
    s_H_d = np.zeros((n1, n2))
    for a in range(nu):
        for i in range(no):
            idet = idx_1[a, i]
            if idet == 0: continue
            I = abs(idet) - 1
            # -h1a(mi) * r1a(am)
            for m in range(no):
                jdet = idx_1[a, m]
                if jdet == 0: continue
                J = abs(jdet) - 1
                s_H_s[I, J] -= fock[o, o][m, i]
            # h1a(ae) * r1a(ei)
            for e in range(nu):
                jdet = idx_1[e, i]
                if jdet == 0: continue
                J = abs(jdet) - 1
                s_H_s[I, J] += fock[v, v][a, e]
            # h2a(amie) * r1a(em)
            for e in range(nu):
                for m in range(no):
                    jdet = idx_1[e, m]
                    if jdet == 0: continue
                    J = abs(jdet) - 1
                    s_H_s[I, J] += g[v, o, o, v][a, m, i, e]
            # h1a(me) * r2a(aeim)
            for e in range(nactu):
                for m in range(no - nacto, no):
                    jdet = idx_2[a, e, i, m]
                    if jdet != 0:
                        J = abs(jdet) - 1
                        phase = np.sign(jdet)
                        s_H_d[I, J] += fock[o, v][m, e] * phase
            # -1/2 h2a(mnif) * r2a(afmn)
            for m in range(no - nacto, no):
                for n in range(m + 1, no):
                    for f in range(nactu):
                        jdet = idx_2[a, f, m, n]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            s_H_d[I, J] -= g[o, o, o, v][m, n, i, f] * phase
            # 1/2 h2a(anef) * r2a(efin)
            for e in range(nactu):
                for f in range(e + 1, nactu):
                    for n in range(no - nacto, no):
                        jdet = idx_2[e, f, i, n]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            s_H_d[I, J] += g[v, o, v, v][a, n, e, f] * phase
    ###########
    # DOUBLES #
    ###########
    d_H_s = np.zeros((n2, n1))
    d_H_d = np.zeros((n2, n2))
    for a in range(nactu):
        for b in range(a + 1, nactu):
            for i in range(no - nacto, no):
                for j in range(i + 1, no):
                    idet = idx_2[a, b, i, j]
                    if idet == 0: continue
                    I = abs(idet) - 1
                    # -A(ab) h2a(amij) * r1a(bm)
                    for m in range(no):
                        # (1)
                        jdet = idx_1[b, m]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            d_H_s[I, J] -= g[v, o, o, o][a, m, i, j]
                        # (ab)
                        jdet = idx_1[a, m]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            d_H_s[I, J] += g[v, o, o, o][b, m, i, j]
                    # A(ij) h2a(abie) * r1a(ej)
                    for e in range(nu):
                        # (1)
                        jdet = idx_1[e, i]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            d_H_s[I, J] += g[v, v, o, v][b, a, j, e]
                        # (ij)
                        jdet = idx_1[e, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            d_H_s[I, J] -= g[v, v, o, v][b, a, i, e]
                    # -A(ij) h1a(mi) * r2a(abmj)
                    for m in range(no - nacto, no):
                        # (1)
                        jdet = idx_2[a, b, m, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            d_H_d[I, J] -= fock[o, o][m, i] * phase
                        # (ij)
                        jdet = idx_2[a, b, m, i]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            d_H_d[I, J] += fock[o, o][m, j] * phase
                    # A(ab) h1a(ae) * r2a(ebij)
                    for e in range(nactu):
                        # (1)
                        jdet = idx_2[e, b, i, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            d_H_d[I, J] += fock[v, v][a, e] * phase
                        # (ab)
                        jdet = idx_2[e, a, i, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            d_H_d[I, J] -= fock[v, v][b, e] * phase
                    # 1/2 h2a(mnij) * r2a(abmn)
                    for m in range(no - nacto, no):
                        for n in range(m + 1, no):
                            jdet = idx_2[a, b, m, n]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                d_H_d[I, J] += g[o, o, o, o][m, n, i, j] * phase
                    # 1/2 h2a(abef) * r2a(efij)
                    for e in range(nactu):
                        for f in range(e + 1, nactu):
                            jdet = idx_2[e, f, i, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                d_H_d[I, J] += g[v, v, v, v][a, b, e, f] * phase
                    # A(ij)A(ab) h2a(amie) * r2a(ebmj)
                    for e in range(nactu):
                        for m in range(no - nacto, no):
                            # (1)
                            jdet = idx_2[e, b, m, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                d_H_d[I, J] += g[v, o, o, v][a, m, i, e] * phase
                            # (ij)
                            jdet = idx_2[e, b, m, i]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                d_H_d[I, J] -= g[v, o, o, v][a, m, j, e] * phase
                            # (ab)
                            jdet = idx_2[e, a, m, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                d_H_d[I, J] -= g[v, o, o, v][b, m, i, e] * phase
                            # (ij)(ab)
                            jdet = idx_2[e, a, m, i]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                d_H_d[I, J] += g[v, o, o, v][b, m, j, e] * phase

    # Assemble and return full matrix
    return np.concatenate((np.concatenate((s_H_s, s_H_d), axis=1),
                           np.concatenate((d_H_s, d_H_d), axis=1),), axis=0)

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

def build_2p_hamiltonian(f, g, o, v, nactu):
    # get orbital parameters
    no, nu = f[o, v].shape
    # set active space parameters
    nactu = min(nactu, nu)
    # allocate active-space 2p hamiltonian
    ndim = int(nactu * (nactu - 1) / 2)
    H = np.zeros((ndim, ndim))
    ct1 = 0
    for a in range(nactu):
        for b in range(a + 1, nactu):
            ct2 = 0
            for c in range(nactu):
                for d in range(c + 1, nactu):
                    H[ct1, ct2] = (
                        +(b == d) * f[v, v][a, c]
                        -(a == d) * f[v, v][b, c]
                        -(b == c) * f[v, v][a, d]
                        +(a == c) * f[v, v][b, d]
                        + g[v, v, v, v][a, b, c, d]
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


def get_index_arrays(no, nu, nacto, nactu):

    nacto = min(no, nacto)
    nactu = min(nu, nactu)

    idx_1 = np.zeros((nu, no), dtype=np.int32)
    kout = 1
    for a in range(nu):
        for i in range(no):
            idx_1[a, i] = kout
            kout += 1
    idx_2 = np.zeros((nu, nu, no, no), dtype=np.int32)
    kout = 1
    for a in range(nactu):
        for b in range(a + 1, nactu):
            for i in range(no - nacto, no):
                for j in range(i + 1, no):
                    idx_2[a, b, i, j] = kout
                    idx_2[b, a, i, j] = -kout
                    idx_2[a, b, j, i] = -kout
                    idx_2[b, a, j, i] = kout
                    kout += 1
    return idx_1, idx_2
