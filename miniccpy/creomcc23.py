import numpy as np
from miniccpy.hbar_diagonal import get_3body_hbar_triples_diagonal

def kernel(T, R, L, r0, omega, fock, H1, H2, o, v):

    t1, t2 = T
    r1, r2 = R
    l1, l2 = L
    # get 3-body Hbar triples diagonal
    d3v, d3o = get_3body_hbar_triples_diagonal(H2[o, o, v, v], t2)

    # Intermediates
    I_vooo = H2[v, o, o, o] - np.einsum("me,aeij->amij", H1[o, v], t2, optimize=True)

    X_ov = np.einsum("mnef,fn->me", H2[o, o, v, v], r1, optimize=True)
    X_vvov =(
        np.einsum("amje,bm->baje", H2[v, o, o, v], r1, optimize=True)
        + np.einsum("amfe,bejm->bajf", H2[v, o, v, v], r2, optimize=True)
        + 0.5 * np.einsum("abfe,ej->bajf", H2[v, v, v, v], r1, optimize=True)
        + 0.25 * np.einsum("nmje,abmn->baje", H2[o, o, o, v], r2, optimize=True)
        - 0.5 * np.einsum("me,abmj->baje", X_ov, t2, optimize=True) 
    )
    X_vvov -= np.transpose(X_vvov, (1, 0, 2, 3))

    X_vooo = (
        -np.einsum("bmie,ej->bmji", H2[v, o, o, v], r1, optimize=True)
        +np.einsum("nmie,bejm->bnji", H2[o, o, o, v], r2, optimize=True)
        - 0.5 * np.einsum("nmij,bm->bnji", H2[o, o, o, o], r1, optimize=True)
        + 0.25 * np.einsum("bmfe,efij->bmji", H2[v, o, v, v], r2, optimize=True)
    )
    X_vooo -= np.transpose(X_vooo, (0, 1, 3, 2))
    # Ground-state moment
    M3 = r0 * (9.0 / 36.0) * (
             np.einsum("abie,ecjk->abcijk", H2[v, v, o, v], t2, optimize=True)
            -np.einsum("amij,bcmk->abcijk", I_vooo, t2, optimize=True)
    )
    # Excited-state moment
    M3 += 0.25 * np.einsum("baje,ecik->abcijk", X_vvov, t2, optimize=True)
    M3 += 0.25 * np.einsum("baje,ecik->abcijk", H2[v, v, o, v], r2, optimize=True)
    M3 -= 0.25 * np.einsum("bmji,acmk->abcijk", X_vooo, t2, optimize=True)
    M3 -= 0.25 * np.einsum("bmji,acmk->abcijk", H2[v, o, o, o], r2, optimize=True)
    M3 -= np.transpose(M3, (0, 1, 2, 3, 5, 4)) # (jk)
    M3 -= np.transpose(M3, (0, 1, 2, 4, 3, 5)) + np.transpose(M3, (0, 1, 2, 5, 4, 3)) # (i/jk)
    M3 -= np.transpose(M3, (0, 2, 1, 3, 4, 5)) # (bc)
    M3 -= np.transpose(M3, (2, 1, 0, 3, 4, 5)) + np.transpose(M3, (1, 0, 2, 3, 4, 5)) # (a/bc)

    # Left 
    L3 = (9.0 / 36.0) * (
            np.einsum("ijab,ck->abcijk", H2[o, o, v, v], l1, optimize=True)
           +np.einsum("ia,bcjk->abcijk", H1[o, v], l2, optimize=True)
           +np.einsum("eiba,ecjk->abcijk", H2[v, o, v, v], l2, optimize=True)
           -np.einsum("jima,bcmk->abcijk", H2[o, o, o, v], l2, optimize=True)
    )
    L3 -= np.transpose(L3, (0, 1, 2, 3, 5, 4)) # (jk)
    L3 -= np.transpose(L3, (0, 1, 2, 4, 3, 5)) + np.transpose(L3, (0, 1, 2, 5, 4, 3)) # (i/jk)
    L3 -= np.transpose(L3, (0, 2, 1, 3, 4, 5)) # (bc)
    L3 -= np.transpose(L3, (2, 1, 0, 3, 4, 5)) + np.transpose(L3, (1, 0, 2, 3, 4, 5)) # (a/bc)

    # Compute triples correction in loop
    delta_A = 0.0
    delta_B = 0.0
    delta_C = 0.0
    delta_D = 0.0
    nu, no = t1.shape
    for a in range(nu):
        for b in range(a + 1, nu):
            for c in range(b + 1, nu):
                for i in range(no):
                    for j in range(i + 1, no):
                        for k in range(j + 1, no):
                            LM = L3[a, b, c, i, j, k] * M3[a, b, c, i, j, k]
                            # A correction
                            dA = (
                                        fock[o, o][i, i] + fock[o, o][j, j] + fock[o, o][k, k]
                                        - fock[v, v][a, a] - fock[v, v][b, b] - fock[v, v][c, c]
                            )
                            delta_A += LM/(omega + dA)
                            # B correction
                            dB = (
                                        H1[o, o][i, i] + H1[o, o][j, j] + H1[o, o][k, k]
                                        - H1[v, v][a, a] - H1[v, v][b, b] - H1[v, v][c, c]
                            )
                            delta_B += LM/(omega + dB)
                            # C correction
                            dC = dB + (
                                    -H2[v, o, o, v][a, i, i, a] - H2[v, o, o, v][b, i, i, b] - H2[v, o, o, v][c, i, i, c]
                                    -H2[v, o, o, v][a, j, j, a] - H2[v, o, o, v][b, j, j, b] - H2[v, o, o, v][c, j, j, c]
                                    -H2[v, o, o, v][a, k, k, a] - H2[v, o, o, v][b, k, k, b] - H2[v, o, o, v][c, k, k, c]
                                    -H2[o, o, o, o][j, i, j, i] - H2[o, o, o, o][k, i, k, i] - H2[o, o, o, o][k, j, k, j]
                                    -H2[v, v, v, v][b, a, b, a] - H2[v, v, v, v][c, a, c, a] - H2[v, v, v, v][c, b, c, b]
                            )
                            delta_C += LM/(omega + dC)
                            # D correction
                            dD = dC + (
                                    +d3o[a, i, j] + d3o[a, i, k] + d3o[a, j, k]
                                    +d3o[b, i, j] + d3o[b, i, k] + d3o[b, j, k]
                                    +d3o[c, i, j] + d3o[c, i, k] + d3o[c, j, k]
                                    -d3v[a, i, b] - d3v[a, i, c] - d3v[b, i, c]
                                    -d3v[a, j, b] - d3v[a, j, c] - d3v[b, j, c]
                                    -d3v[a, k, b] - d3v[a, k, c] - d3v[b, k, c]
                            )
                            delta_D += LM/(omega + dD)

    # Store triples corrections in dictionary
    delta_T = {"A": delta_A, "B": delta_B, "C": delta_C, "D": delta_D}
    return delta_T
