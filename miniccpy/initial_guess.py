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
    omega = np.real(omega[idx])
    C_act = np.real(C_act[:, idx])

    nroot = min(nroot, C_act.shape[1])
    no, nu = f[o, v].shape
    ndim = no*nu + no**2*nu**2
    C = np.zeros((ndim, nroot))

    # For closed shells, we can pick out singlets and triplets numerically
    if is_closed_shell and mult != -1:
        omega_guess = np.zeros(nroot)
        C_spin = np.zeros((C.shape[0], nroot))
        n_spin = 0
        for n in range(C.shape[1]):
            if spin_function1(C_act[:no*nu, n], mult, no, nu) <= 1.0e-06:
                C_spin[:, n_spin] = cisd_scatter(C_act[:, n], nacto, nactu, no, nu)
                omega_guess[n_spin] = omega[n]
                n_spin += 1
            if n_spin >= nroot:
                break
        else:
            print(f"    Could not find {nroot} roots with mult = {mult}. Returning {n_spin} found roots.\n")
        # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
        R_guess, _ = np.linalg.qr(C_spin[:, :min(n_spin, nroot)])
        omega_guess = omega_guess[:min(n_spin, nroot)]
    else:
        for i in range(nroot):
            C[:, i] = cisd_scatter(C_act[:, i], nacto, nactu, no, nu)
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
    omega = np.real(omega[idx])
    C = np.real(C[:, idx])

    # For closed shells, we can pick out singlets and triplets numerically
    if is_closed_shell and mult != -1:
        omega_guess = np.zeros(nroot)
        C_spin = np.zeros((C.shape[0], nroot))
        n_spin = 0
        for n in range(C.shape[1]):
            if spin_function1(C[:, n], mult, no, nu) <= 1.0e-06:
                C_spin[:, n_spin] = C[:, n]
                omega_guess[n_spin] = omega[n]
                n_spin += 1
            if n_spin >= nroot:
                break
        else:
            print(f"    Could not find {nroot} roots with mult = {mult}. Returning {n_spin} found roots.\n")
        # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
        R_guess, _ = np.linalg.qr(C_spin[:, :min(n_spin, nroot)])
        omega_guess = omega_guess[:min(n_spin, nroot)]
    else:
        # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
        R_guess, _ = np.linalg.qr(C[:, :nroot])
        omega_guess = omega[:nroot]

    return R_guess, omega_guess

def rcisd_guess(f, g, o, v, nroot, nacto, nactu, mult=1):
    """Obtain the lowest `nroot` roots of the RHF CISd Hamiltonian
    to serve as the initial guesses for the EOMCC calculations."""

    nu, no = f[v, o].shape
    nacto = min(nacto, no)
    nactu = min(nactu, nu)
    # print dimensions of initial guess procedure
    print("   CISd initial guess")
    print("   Multiplicity = ", mult)
    print("   Number of roots = ", nroot)
    print("   Dimension of eigenvalue problem = ", no*nu + (nacto + 1)*nacto/2 * (nactu + 1)*nactu/2)
    print("   Active occupied = ", nacto)
    print("   Active unoccupied = ", nactu)
    print("   -----------------------------------")

    # Build the CISd Hamiltonian and diagonalize
    H = build_rcisd_hamiltonian(f, g, o, v, nacto, nactu)
    omega, C_act = np.linalg.eig(H)
    idx = np.argsort(omega)
    omega = np.real(omega[idx])
    C_act = np.real(C_act[:, idx])

    nroot = min(nroot, C_act.shape[1])
    no, nu = f[o, v].shape
    ndim = no*nu + no**2*nu**2
    C = np.zeros((ndim, nroot))

    for i in range(nroot):
        C[:, i] = rcisd_scatter(C_act[:, i], nacto, nactu, no, nu)
    # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
    R_guess, _ = np.linalg.qr(C[:, :nroot])
    omega_guess = omega[:nroot]

    return R_guess, omega_guess

def rcis_guess(f, g, o, v, nroot, mult=1):
    """Obtain the lowest `nroot` roots of the RCIS Hamiltonian
    to serve as the initial guesses for the RHF-EOMCC calculations."""

    nu, no = f[v, o].shape
    # print dimensions of initial guess procedure
    print("   RCIS initial guess")
    print("   Multiplicity = ", mult)
    print("   Number of roots = ", nroot)
    print("   Dimension of eigenvalue problem = ", no*nu)
    print("   -----------------------------------")

    H = build_rcis_hamiltonian(f, g, o, v)
    omega, C = np.linalg.eig(H)
    idx = np.argsort(omega)
    omega = np.real(omega[idx])
    C = np.real(C[:, idx])
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
    omega = np.real(omega[idx])
    C = np.real(C[:, idx])

    no, nu = f[o, v].shape
    n1 = nu**2

    R_guess = np.zeros((n1, nroot))
    for i in range(nroot):
        R_guess[:, i] = deacis_scatter(C[:, i], nactu, no, nu)

    # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
    R_guess, _ = np.linalg.qr(R_guess)

    return R_guess, omega[:nroot]

def dipcis_guess(f, g, o, v, nroot):
    """Obtain the lowest `nroot` roots of the 2h Hamiltonian
    to serve as the initial guesses for the DIP-EOMCC calculations."""
    H = build_2h_hamiltonian(f, g, o, v)
    omega, C = np.linalg.eig(H)
    idx = np.argsort(omega)
    omega = np.real(omega[idx])
    C = np.real(C[:, idx])

    no, nu = f[o, v].shape
    n1 = no**2

    R_guess = np.zeros((n1, nroot))
    for i in range(nroot):
        R_guess[:, i] = dipcis_scatter(C[:, i], no)
        
    # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
    R_guess, _ = np.linalg.qr(R_guess[:, :nroot])

    return R_guess, omega[:nroot]

def dipcis_cvs_guess(f, g, o, v, nroot, cvsmin, cvsmax):
    """Obtain the lowest `nroot` roots of the 2h Hamiltonian
    to serve as the initial guesses for the DIP-EOMCC calculations."""

    H = build_2h_cvs_hamiltonian(f, g, o, v, cvsmin, cvsmax)
    omega, C = np.linalg.eig(H)
    idx = np.argsort(omega)[::-1]
    omega = np.real(omega[idx])
    C = np.real(C[:, idx])

    no, nu = f[o, v].shape
    n1 = no**2

    R_guess = np.zeros((n1, nroot))
    for i in range(nroot):
        R_guess_i = dipcis_scatter(C[:, i], no).copy().reshape(no, no)
        # apply CVS separation to the guess vector
        for j in range(no):
            for k in range(j + 1, no):
                if (j < cvsmin or j > cvsmax) and (k < cvsmin or k > cvsmax):
                    R_guess_i[j, k] = 0.0
                    R_guess_i[k, j] = 0.0
        R_guess[:, i] = R_guess_i.flatten()
    # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
    R_guess, _ = np.linalg.qr(R_guess[:, :nroot])
    return R_guess, omega[:nroot]

def dipcisd_guess(f, g, o, v, nroot, nacto=0, nactu=0):
    """Obtain the lowest `nroot` roots of the 2h + active 3h1p Hamiltonian
    to serve as the initial guesses for the DIP-EOMCC calculations."""
    H = build_dipcisd_hamiltonian(f, g, o, v, nacto, nactu)
    omega, C = np.linalg.eig(H)
    idx = np.argsort(omega)
    omega = np.real(omega[idx])
    C = np.real(C[:, idx])

    no, nu = f[o, v].shape
    n1 = no**2
    n2 = no**3 * nu

    R_guess = np.zeros((n1 + n2, nroot))
    for i in range(nroot):
        R_guess[:, i] = dipcisd_scatter(C[:, i], no, nu, nacto, nactu)

    # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
    R_guess, _ = np.linalg.qr(R_guess[:, :nroot])
    return R_guess, omega[:nroot]

def dipcisd_cvs_guess(f, g, o, v, nroot, cvsmin, cvsmax, nacto=0, nactu=0):
    """Obtain the lowest `nroot` roots of the 2h + active 3h1p Hamiltonian
    to serve as the initial guesses for the DIP-EOMCC calculations."""
    H = build_dipcisd_hamiltonian(f, g, o, v, nacto, nactu)
    omega, C = np.linalg.eig(H)
    idx = np.argsort(omega)[::-1]
    omega = np.real(omega[idx])
    C = np.real(C[:, idx])

    no, nu = f[o, v].shape
    n1 = no**2
    n2 = no**3 * nu

    R_guess = np.zeros((n1 + n2, nroot))
    for i in range(nroot):
        R_guess_i = dipcisd_scatter(C[:, i], no, nu, nacto, nactu)
        r1 = R_guess_i[:n1].reshape(no, no)
        r2 = R_guess_i[n1:].reshape(no, no, nu, no)
        # apply CVS separation to the guess vector
        for j in range(no):
            for k in range(j + 1, no):
                if (j < cvsmin or j > cvsmax) and (k < cvsmin or k > cvsmax):
                    r1[j, k] = 0.0
                    r1[k, j] = 0.0
        for j in range(no):
            for k in range(j + 1, no):
                for l in range(k + 1, no):
                    for d in range(nu):
                        if (j < cvsmin or j > cvsmax) and (k < cvsmin or k > cvsmax) and (l < cvsmin or l > cvsmax):
                            r2[j, k, d, l] = 0.0
                            r2[j, l, d, k] = 0.0
                            r2[k, j, d, l] = 0.0
                            r2[k, l, d, j] = 0.0
                            r2[l, j, d, k] = 0.0
                            r2[l, k, d, j] = 0.0
        R_guess[:, i] = np.hstack((r1.flatten(), r2.flatten()))
    # orthonormalize the initial trial space; this is important when using doubles in EOMCCSd guess
    R_guess, _ = np.linalg.qr(R_guess[:, :nroot])
    return R_guess, omega[:nroot]

def eacis_guess(f, g, o, v, nroot):
    """Obtain the lowest `nroot` roots of the 1p Hamiltonian
    to serve as the initial guesses for the EA-EOMCC calculations."""

    H = build_1p_hamiltonian(f, g, o, v)
    omega, C = np.linalg.eig(H)
    idx = np.argsort(omega)
    omega = np.real(omega[idx])
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
    omega = np.real(omega[idx])
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

def rcisd_scatter(V_in, nacto, nactu, no, nu):

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
        for b in range(a, nu):
            for i in range(no):
                for j in range(i, no):
                    if a < nactu and b < nactu and i >= no - nacto and j >= no - nacto:
                        V2_out[a, b, i, j] = V_in[offset]
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

def dipcisd_scatter(V_in, no, nu, nacto, nactu):

    # allocate full-length output vector
    V1_out = np.zeros((no, no))
    V2_out = np.zeros((no, no, nu, no))
    # Start filling in the array
    offset = 0
    for i in range(no):
        for j in range(i + 1, no):
            V1_out[i, j] = V_in[offset]
            V1_out[j, i] = -V_in[offset]
            offset += 1
    for i in range(no):
        for j in range(i + 1, no):
            for c in range(nu):
                for k in range(j + 1, no):
                    if c < nactu and i >= no - nacto and j >= no - nacto and k >= no - nacto:
                        V2_out[i, j, c, k] = V_in[offset]
                        V2_out[i, k, c, j] = -V_in[offset]
                        V2_out[j, i, c, k] = -V_in[offset]
                        V2_out[j, k, c, i] = V_in[offset]
                        V2_out[k, i, c, j] = V_in[offset]
                        V2_out[k, j, c, i] = -V_in[offset]
                        offset += 1
    return np.hstack((V1_out.flatten(), V2_out.flatten()))

# def cvs_dipcisd_scatter(V_in, no, nu, cvsmin, cvsmax, nactu):
#
#     # allocate full-length output vector
#     V1_out = np.zeros((no, no))
#     V2_out = np.zeros((no, no, nu, no))
#     # Start filling in the array
#     offset = 0
#     for i in range(no):
#         for j in range(i + 1, no):
#             if (i < cvsmin or i > cvsmax) and (j < cvsmin or j > cvsmax): continue
#             V1_out[i, j] = V_in[offset]
#             V1_out[j, i] = -V_in[offset]
#             offset += 1
#     for i in range(no):
#         for j in range(i + 1, no):
#             for c in range(nu):
#                 for k in range(j + 1, no):
#                     if (i < cvsmin or i > cvsmax) and (j < cvsmin or j > cvsmax) and (k < cvsmin or k > cvsmax): continue
#                     if c < nactu:
#                         V2_out[i, j, c, k] = V_in[offset]
#                         V2_out[i, k, c, j] = -V_in[offset]
#                         V2_out[j, i, c, k] = -V_in[offset]
#                         V2_out[j, k, c, i] = V_in[offset]
#                         V2_out[k, i, c, j] = V_in[offset]
#                         V2_out[k, j, c, i] = -V_in[offset]
#                         offset += 1
#     return np.hstack((V1_out.flatten(), V2_out.flatten()))

def dipcis_scatter(V_in, no):

    # allocate full-length output vector
    V1_out = np.zeros((no, no))
    # Start filling in the array
    offset = 0
    for i in range(no):
        for j in range(i + 1, no):
            V1_out[i, j] = V_in[offset]
            V1_out[j, i] = -V_in[offset]
            offset += 1
    return V1_out.flatten()

def build_cisd_hamiltonian(fock, g, o, v, nacto, nactu):

    no, nu = fock[o, v].shape
    # set dimensions of CISD problem
    n1 = no * nu
    n2 = int(nacto * (nacto - 1) / 2 * nactu * (nactu - 1) / 2)
    # get index addressing arrays
    idx_1, idx_2 = get_cisd_index_arrays(no, nu, nacto, nactu)
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

def build_rcis_hamiltonian(f, g, o, v):
    """ Construct the RCIS Hamiltonian.
    """

    nu, no = f[v, o].shape
    n1 = no * nu

    idx = np.zeros((nu, no), dtype=np.int32)
    kout = 1
    for a in range(nu):
        for i in range(no):
            idx[a, i] = kout
            kout += 1

    H = np.zeros((n1, n1))
    for a in range(nu):
        for i in range(no):
            idet = idx[a, i]
            if idet == 0: continue
            I = abs(idet) - 1
            # -h1a(mi) * r1a(am)
            for m in range(no):
                jdet = idx[a, m]
                if jdet == 0: continue
                J = abs(jdet) - 1
                H[I, J] -= f[o, o][m, i]
            # h1a(ae) * r1a(ei)
            for e in range(nu):
                jdet = idx[e, i]
                if jdet == 0: continue
                J = abs(jdet) - 1
                H[I, J] += f[v, v][a, e]
            # h2a(amie) * r1a(em)
            for e in range(nu):
                for m in range(no):
                    jdet = idx[e, m]
                    if jdet == 0: continue
                    J = abs(jdet) - 1
                    H[I, J] += g[v, o, o, v][a, m, i, e] - g[v, o, v, o][a, m, e, i]
            # h2b(amie) * r1b(em)
            for e in range(nu):
                for m in range(no):
                    jdet = idx[e, m]
                    if jdet == 0: continue
                    J = abs(jdet) - 1
                    H[I, J] += g[v, o, o, v][a, m, i, e]
    return H

def build_rcisd_hamiltonian(fock, g, o, v, nacto, nactu):

    no, nu = fock[o, v].shape
    # set dimensions of CISD problem
    n1 = no * nu
    n2 = int(nacto * (nacto + 1) / 2 * nactu * (nactu + 1) / 2)
    # get index addressing arrays
    idx_1, idx_2 = get_rcisd_index_arrays(no, nu, nacto, nactu)
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
                    s_H_s[I, J] += g[v, o, o, v][a, m, i, e] - g[v, o, v, o][a, m, e, i]
            # h1b(me) * r2b(aeim)
            for e in range(nactu):
                for m in range(no - nacto, no):
                    jdet = idx_2[a, e, i, m]
                    if jdet != 0:
                        J = abs(jdet) - 1
                        s_H_d[I, J] += fock[o, v][m, e]
            # -h2b(mnif) * r2b(afmn)
            for m in range(no - nacto, no):
                for n in range(no - nacto, no):
                    for f in range(nactu):
                        jdet = idx_2[a, f, m, n]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            s_H_d[I, J] -= g[o, o, o, v][m, n, i, f]
            # h2b(anef) * r2b(efin)
            for e in range(nactu):
                for f in range(nactu):
                    for n in range(no - nacto, no):
                        jdet = idx_2[e, f, i, n]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            s_H_d[I, J] += g[v, o, v, v][a, n, e, f]
    ###########
    # DOUBLES #
    ###########
    d_H_s = np.zeros((n2, n1))
    d_H_d = np.zeros((n2, n2))
    for a in range(nactu):
        for b in range(nactu):
            for i in range(no - nacto, no):
                for j in range(no - nacto, no):
                    idet = idx_2[a, b, i, j]
                    if idet == 0: continue
                    I = abs(idet) - 1
                    # -h2b(mbij) * r1a(am)
                    for m in range(no):
                        jdet = idx_1[a, m]
                        if jdet == 0: continue
                        J = abs(jdet) - 1
                        d_H_s[I, J] -= g[o, v, o, o][m, b, i, j]
                    # h2b(abej) * r1a(ei)
                    for e in range(nu):
                        jdet = idx_1[e, i]
                        if jdet == 0: continue
                        J = abs(jdet) - 1
                        d_H_s[I, J] += g[v, v, v, o][a, b, e, j]
                    # -h2b(amij) * r1b(bm)
                    for m in range(no):
                        jdet = idx_1[b, m]
                        if jdet == 0: continue
                        J = abs(jdet) - 1
                        d_H_s[I, J] -= g[v, o, o, o][a, m, i, j]
                    # h2b(abie) * r1b(ej)
                    for e in range(nu):
                        jdet = idx_1[e, j]
                        if jdet == 0: continue
                        J = abs(jdet) - 1
                        d_H_s[I, J] += g[v, v, o, v][a, b, i, e]
                    # -h1a(mi) * r2b(abmj)
                    for m in range(no - nacto, no):
                        jdet = idx_2[a, b, m, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            d_H_d[I, J] -= fock[o, o][m, i]
                    # -h1b(mj) * r2b(abim)
                    for m in range(no - nacto, no):
                        jdet = idx_2[a, b, i, m]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            d_H_d[I, J] -= fock[o, o][m, j]
                    # h1a(ae) * r2b(ebij)
                    for e in range(nactu):
                        jdet = idx_2[e, b, i, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            d_H_d[I, J] += fock[v, v][a, e]
                    # h1a(be) * r2b(aeij)
                    for e in range(nactu):
                        jdet = idx_2[a, e, i, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            d_H_d[I, J] += fock[v, v][b, e]
                    # h2b(mnij) * r2b(abmn)
                    for m in range(no - nacto, no):
                        for n in range(no - nacto, no):
                            jdet = idx_2[a, b, m, n]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                d_H_d[I, J] += g[o, o, o, o][m, n, i, j]
                    # h2b(abef) * r2b(efij)
                    for e in range(nactu):
                        for f in range(nactu):
                            jdet = idx_2[e, f, i, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                d_H_d[I, J] += g[v, v, v, v][a, b, e, f]
                    # h2a(amie) * r2b(ebmj)
                    for e in range(nactu):
                        for m in range(no - nacto, no):
                            jdet = idx_2[e, b, m, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                d_H_d[I, J] += g[v, o, o, v][a, m, i, e] - g[v, o, v, o][a, m, e, i]
                    # h2c(bmje) * r2b(aeim)
                    for e in range(nactu):
                        for m in range(no - nacto, no):
                            jdet = idx_2[a, e, i, m]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                d_H_d[I, J] += g[v, o, o, v][b, m, j, e] - g[v, o, v, o][b, m, e, j]
                    # -h2b(amej) * r2b(ebim)
                    for e in range(nactu):
                        for m in range(no - nacto, no):
                            jdet = idx_2[e, b, i, m]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                d_H_d[I, J] -= g[v, o, v, o][a, m, e, j]
                    # -h2b(mbie) * r2b(aemj)
                    for e in range(nactu):
                        for m in range(no - nacto, no):
                            jdet = idx_2[a, e, m, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                d_H_d[I, J] -= g[o, v, o, v][m, b, i, e]

    # Assemble and return full matrix
    return np.concatenate((np.concatenate((s_H_s, s_H_d), axis=1),
                           np.concatenate((d_H_s, d_H_d), axis=1),), axis=0)

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

def build_2h_hamiltonian(f, g, o, v):
    """ Construct the 2h Hamiltonian with matrix elements
        given by:
        < ij | H_N | kl > = A(ij)A(kl)[d(i,k)f(l,j)] + g(k,l,i,j) 
    """

    # get orbital parameters
    no, nu = f[o, v].shape
    # allocate active-space 2p hamiltonian
    ndim = int(no * (no - 1) / 2)
    H = np.zeros((ndim, ndim))
    ct1 = 0
    for i in range(no):
        for j in range(i + 1, no):
            ct2 = 0
            for k in range(no):
                for l in range(k + 1, no):
                    H[ct1, ct2] = (
                        -(j == l) * f[o, o][k, i]
                        +(i == l) * f[o, o][k, j]
                        +(j == k) * f[o, o][l, i]
                        -(i == k) * f[o, o][l, j]
                        + g[o, o, o, o][k, l, i, j]
                    )
                    ct2 += 1
            ct1 += 1
    return H

def build_2h_cvs_hamiltonian(f, g, o, v, cvsmin, cvsmax):
    """ Construct the 2h Hamiltonian with matrix elements
        given by:
        < ij | H_N | kl > = A(ij)A(kl)[d(i,k)f(l,j)] + g(k,l,i,j) 
    """
    # get orbital parameters
    no, nu = f[o, v].shape
    # allocate active-space 2p hamiltonian
    ndim = int(no * (no - 1) / 2)
    H = np.zeros((ndim, ndim))
    ct1 = 0
    for i in range(no):
        for j in range(i + 1, no):
            if (i < cvsmin or i > cvsmax) and (j < cvsmin or j > cvsmax): continue
            ct2 = 0
            for k in range(no):
                for l in range(k + 1, no):
                    if (k < cvsmin or k > cvsmax) and (l < cvsmin or l > cvsmax): continue
                    H[ct1, ct2] = (
                        -(j == l) * f[o, o][k, i]
                        +(i == l) * f[o, o][k, j]
                        +(j == k) * f[o, o][l, i]
                        -(i == k) * f[o, o][l, j]
                        + g[o, o, o, o][k, l, i, j]
                    )
                    ct2 += 1
            ct1 += 1
    return H

def build_dipcisd_hamiltonian(fock, g, o, v, nacto, nactu):
    """ Construct the 2h + active(3h-1p) Hamiltonian with matrix elements"""
    no, nu = fock[o, v].shape
    n1 = int(no*(no - 1)/2)
    n2 = int(nacto*(nacto - 1)*(nacto - 2)/6 * nactu)
    # get index addressing arrays
    idx_1, idx_2 = get_dipcisd_index_arrays(no, nu, nacto, nactu)
    ######
    # 2h #
    ######
    s_H_s = np.zeros((n1, n1))
    s_H_d = np.zeros((n1, n2))
    for i in range(no):
        for j in range(i + 1, no):
            idet = idx_1[i, j]
            if idet == 0: continue
            I = abs(idet) - 1
            # -A(ij) h1(mi)*r1(mj)
            for m in range(no):
                # (1)
                jdet = idx_1[m, j]
                if jdet != 0:
                    J = abs(jdet) - 1
                    phase = np.sign(jdet)
                    s_H_s[I, J] -= fock[o, o][m, i] * phase
                # (ij)
                jdet = idx_1[m, i]
                if jdet != 0:
                    J = abs(jdet) - 1
                    phase = np.sign(jdet)
                    s_H_s[I, J] += fock[o, o][m, j] * phase
            # h2(mnij)*r1(mn)
            for m in range(no):
                for n in range(m + 1, no):
                    # (1)
                    jdet = idx_1[m, n]
                    if jdet != 0:
                        J = abs(jdet) - 1
                        phase = np.sign(jdet)
                        s_H_s[I, J] += g[o, o, o, o][m, n, i, j] * phase
            # h1(me)*r2(ijem)
            for e in range(nactu):
                for m in range(no - nacto, no):
                    # (1)
                    jdet = idx_2[i, j, e, m]
                    if jdet != 0:
                        J = abs(jdet) - 1
                        phase = np.sign(jdet)
                        s_H_d[I, J] += fock[o, v][m, e] * phase
            # -A(ij) h2(mnif)*r2(mjfn)
            for m in range(no - nacto, no):
                for n in range(m + 1, no):
                    for f in range(nactu):
                        # (1)
                        jdet = idx_2[m, j, f, n]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            s_H_d[I, J] -= g[o, o, o, v][m, n, i, f] * phase
                        # (ij)
                        jdet = idx_2[m, i, f, n]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            s_H_d[I, J] += g[o, o, o, v][m, n, j, f] * phase
    ########
    # 3h1p #
    ########
    d_H_s = np.zeros((n2, n1))
    d_H_d = np.zeros((n2, n2))
    for i in range(no - nacto, no):
        for j in range(i + 1, no):
            for c in range(nactu):
                for k in range(j + 1, no):
                    idet = idx_2[i, j, c, k]
                    if idet == 0: continue
                    I = abs(idet) - 1
                    # -A(j/ik) h2(cmki)*r1(mj)
                    for m in range(no):
                        # (1)
                        jdet = idx_1[m, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            d_H_s[I, J] -= g[v, o, o, o][c, m, k, i] * phase
                        # (ij)
                        jdet = idx_1[m, i]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            d_H_s[I, J] += g[v, o, o, o][c, m, k, j] * phase
                        # (jk)
                        jdet = idx_1[m, k]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            d_H_s[I, J] += g[v, o, o, o][c, m, j, i] * phase
                    # h1(ce)*r2(ijek)
                    for e in range(nactu):
                        jdet = idx_2[i, j, e, k]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            d_H_d[I, J] += fock[v, v][c, e] * phase
                    # -A(k/ij) h1(mk)*r2(ijcm)
                    for m in range(no - nacto, no):
                        # (1)
                        jdet = idx_2[i, j, c, m]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            d_H_d[I, J] -= fock[o, o][m, k] * phase
                        # (ik)
                        jdet = idx_2[k, j, c, m]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            d_H_d[I, J] += fock[o, o][m, i] * phase
                        # (jk)
                        jdet = idx_2[i, k, c, m]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            d_H_d[I, J] += fock[o, o][m, j] * phase
                    # A(k/ij) h2(mnij)*r2(mnck)
                    for m in range(no - nacto, no):
                        for n in range(m + 1, no):
                            # (1)
                            jdet = idx_2[m, n, c, k]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                d_H_d[I, J] += g[o, o, o, o][m, n, i, j] * phase
                            # (ik)
                            jdet = idx_2[m, n, c, i]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                d_H_d[I, J] -= g[o, o, o, o][m, n, k, j] * phase
                            # (jk)
                            jdet = idx_2[m, n, c, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                d_H_d[I, J] -= g[o, o, o, o][m, n, i, k] * phase
                    # A(k/ij) h2(cmke)*r2(ijem)
                    for m in range(no - nacto, no):
                        for e in range(nactu):
                            # (1)
                            jdet = idx_2[i, j, e, m]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                d_H_d[I, J] += g[v, o, o, v][c, m, k, e] * phase
                            # (ik)
                            jdet = idx_2[k, j, e, m]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                d_H_d[I, J] -= g[v, o, o, v][c, m, i, e] * phase
                            # (jk)
                            jdet = idx_2[i, k, e, m]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                d_H_d[I, J] -= g[v, o, o, v][c, m, j, e] * phase
    # Assemble and return full matrix
    return np.concatenate((np.concatenate((s_H_s, s_H_d), axis=1),
                           np.concatenate((d_H_s, d_H_d), axis=1),), axis=0)

def build_1p_hamiltonian(f, g, o, v):
    """ Construct the 1p Hamiltonian with matrix elements
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

def spin_function1(C1, mult, no, nu):
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
                c1_b[(a - 1) // 2, (i - 1) // 2] = c1_arr[a, i]
    # For RHF, singlets have c1_a = c1_b and triplets have c1_a = -c1_b
    if mult == 1:
        error = np.linalg.norm(c1_a - c1_b)
    elif mult == 3:
        error = np.linalg.norm(c1_a + c1_b)
    return error

# def spin_function2(C2, mult, no, nu):
#     # Reshape the excitation vector into C2
#     c2_arr = np.reshape(np.real(C2), (nu, nu, no, no))
#     # Create the a->a and b->b single excitation cases
#     c1_aa = np.zeros((nu // 2, nu // 2, no // 2, no // 2))
#     c1_ab = np.zeros((nu // 2, nu // 2, no // 2, no // 2))
#     c1_bb = np.zeros((nu // 2, nu // 2, no // 2, no // 2))
#     for a in range(nu):
#         for b in range()
#         for i in range(no):
#             if a % 2 == 0 and i % 2 == 0:
#                 c1_a[a // 2, i // 2] = c1_arr[a, i]
#             elif a % 2 == 1 and i % 2 == 1:
#                 c1_b[(a - 1) // 2, (i - 1) // 2] = c1_arr[a, i]
#     # For RHF, singlets have c1_a = c1_b and triplets have c1_a = -c1_b
#     if mult == 1:
#         error = np.linalg.norm(c1_a - c1_b)
#     elif mult == 3:
#         error = np.linalg.norm(c1_a + c1_b)
#     return error

def get_cisd_index_arrays(no, nu, nacto, nactu):

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

def get_rcisd_index_arrays(no, nu, nacto, nactu):

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
        for b in range(a, nactu):
            for i in range(no - nacto, no):
                for j in range(i, no):
                    idx_2[a, b, i, j] = kout
                    idx_2[b, a, j, i] = kout
                    kout += 1
    return idx_1, idx_2

def get_dipcisd_index_arrays(no, nu, nacto, nactu):

    nacto = min(no, nacto)
    nactu = min(nu, nactu)

    idx_1 = np.zeros((no, no), dtype=np.int32)
    kout = 1
    for i in range(no):
        for j in range(i + 1, no):
            idx_1[i, j] = kout
            idx_1[j, i] = -kout
            kout += 1
    idx_2 = np.zeros((no, no, nu, no), dtype=np.int32)
    kout = 1
    for i in range(no - nacto, no):
        for j in range(i + 1, no):
            for c in range(nactu):
                for k in range(j + 1, no):
                    idx_2[i, j, c, k] = kout
                    idx_2[i, k, c, j] = -kout
                    idx_2[j, i, c, k] = -kout
                    idx_2[j, k, c, i] = kout
                    idx_2[k, i, c, j] = kout
                    idx_2[k, j, c, i] = -kout
                    kout += 1
    return idx_1, idx_2

# def get_cvs_dipcisd_index_arrays(no, nu, nactu, cvsmin, cvsmax):
#
#     idx_1 = np.zeros((no, no), dtype=np.int32)
#     kout = 1
#     for i in range(no):
#         for j in range(i + 1, no):
#             if (i < cvsmin or i > cvsmax) and (j < cvsmin or j > cvsmax): continue
#             idx_1[i, j] = kout
#             idx_1[j, i] = -kout
#             kout += 1
#     idx_2 = np.zeros((no, no, nu, no), dtype=np.int32)
#     kout = 1
#     for i in range(no):
#         for j in range(i + 1, no):
#             for c in range(nactu):
#                 for k in range(j + 1, no):
#                     if (i < cvsmin or i > cvsmax) and (j < cvsmin or j > cvsmax) and (k < cvsmin or k > cvsmax): continue
#                     idx_2[i, j, c, k] = kout
#                     idx_2[i, k, c, j] = -kout
#                     idx_2[j, i, c, k] = -kout
#                     idx_2[j, k, c, i] = kout
#                     idx_2[k, i, c, j] = kout
#                     idx_2[k, j, c, i] = -kout
#                     kout += 1
#     return idx_1, idx_2
