import numpy as np

def kernel(T, R, L, r0, omega, H1, H2, o, v):

    t1, t2 = T
    r1, r2 = R
    l1, l2 = L
    eps = np.diagonal(H1)
    n = np.newaxis
    e_abcijk = (omega - eps[v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                      + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o])


    I_vooo = H2[v, o, o, o] - np.einsum("me,aeij->amij", H1[o, v], t2, optimize=True)

    # Moments
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

    M3 = 0.25 * np.einsum("baje,ecik->abcijk", X_vvov, t2, optimize=True)
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
    L3 /= e_abcijk

    return (1.0 / 36.0) * np.einsum("abcijk,abcijk->", L3, M3, optimize=True)
