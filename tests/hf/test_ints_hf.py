import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar

def main():

    basis = '6-31g'
    nfrozen = 0
    # Define molecule geometry and basis set
    geom = [['H', (0.0, 0.0, -0.8)], 
            ['F', (0.0, 0.0,  0.8)]]

    fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, unit="Angstrom")

    T, E_corr  = run_cc_calc(fock, g, o, v, method='cc3')
    assert np.isclose(E_corr, -0.178932834216145, atol=1e-9)

    H1, H2 = get_hbar(T, fock, g, o, v, method='ccsdt')
    R, omega_guess = run_guess(H1, H2, o, v, 5, method="cis", mult=1)
    R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eomcc3", state_index=[0], fock=fock, g=g, denom_type="fock")

    #I_vooo, I_vvov, I_oooo, I_vvvv = compute_cc3_intermediates(g, T[0], o, v)
    x_vooo, x_vvov, x_oooo, x_vvvv, x_voov = compute_eomcc3_intermediates(g, T[0], o, v)
    
    nu, no = T[0].shape
    noa = no // 2
    nua = nu // 2
    nob = no // 2
    nub = nu // 2

    eps = np.diagonal(fock)
    n = np.newaxis
    e_abcijk = -1.0 / (eps[v, n, n, n, n, n] + eps[n, v, n, n, n, n] + eps[n, n, v, n, n, n]
                - eps[n, n, n, o, n, n] - eps[n, n, n, n, o, n] - eps[n, n, n, n, n, o])
    t3 = compute_t3(T[1], x_vooo, x_vvov, e_abcijk)

    # Test T3
    print(np.linalg.norm(t3.flatten() - T[2].flatten()))

    # Test Hbar of CC3 / CCSDT
    # h(vooo)
    h2a_vooo = np.zeros((nua, noa, noa, noa))
    h2b_vooo = np.zeros((nua, nob, noa, nob))
    h2b_ovoo = np.zeros((noa, nub, noa, nob))
    h2c_vooo = np.zeros((nub, nob, nob, nob))
    for a in range(nu):
        for m in range(no):
            for i in range(no):
                for j in range(no):
                    hmatel = H2[v, o, o, o][a, m, i, j]
                    if a % 2 == 0 and m % 2 == 0 and i % 2 == 0 and j % 2 == 0:
                        h2a_vooo[a // 2, m // 2, i // 2, j // 2] = hmatel
                    if a % 2 == 1 and m % 2 == 1 and i % 2 == 1 and j % 2 == 1:
                        h2c_vooo[(a - 1) // 2, (m - 1) // 2, (i - 1) // 2, (j - 1) // 2] = hmatel
                    if a % 2 == 0 and m % 2 == 1 and i % 2 == 0 and j % 2 == 1:
                        h2b_vooo[a // 2, (m - 1) // 2, i // 2, (j - 1) // 2] = hmatel
                    if a % 2 == 1 and m % 2 == 0 and i % 2 == 1 and j % 2 == 0:
                        h2b_ovoo[m // 2, (a - 1) // 2, j // 2, (i - 1) // 2] = hmatel
    print("h2a_vooo = ", np.linalg.norm(h2a_vooo.flatten()))
    print("h2b_vooo = ", np.linalg.norm(h2b_vooo.flatten()))
    print("h2b_ovoo = ", np.linalg.norm(h2b_ovoo.flatten()))
    print("h2c_vooo = ", np.linalg.norm(h2c_vooo.flatten()))
    print("")

    # h(vvov)
    h2a_vvov = np.zeros((nua, nua, noa, nua))
    h2b_vvov = np.zeros((nua, nub, noa, nub))
    h2b_vvvo = np.zeros((nua, nub, nua, nob))
    h2c_vvov = np.zeros((nub, nub, nob, nub))
    for a in range(nu):
        for b in range(nu):
            for i in range(no):
                for e in range(nu):
                    hmatel = H2[v, v, o, v][a, b, i, e]
                    if a % 2 == 0 and b % 2 == 0 and i % 2 == 0 and e % 2 == 0:
                        h2a_vvov[a // 2, b // 2, i // 2, e // 2] = hmatel
                    if a % 2 == 1 and b % 2 == 1 and i % 2 == 1 and e % 2 == 1:
                        h2c_vvov[(a - 1) // 2, (b - 1) // 2, (i - 1) // 2, (e - 1) // 2] = hmatel
                    if a % 2 == 0 and b % 2 == 1 and i % 2 == 0 and e % 2 == 1:
                        h2b_vvov[a // 2, (b - 1) // 2, i // 2, (e - 1) // 2] = hmatel
                    if a % 2 == 1 and b % 2 == 0 and i % 2 == 1 and e % 2 == 0:
                        h2b_vvvo[b // 2, (a - 1) // 2, e // 2, (i - 1) // 2] = hmatel
    print("h2a_vvov = ", np.linalg.norm(h2a_vvov.flatten()))
    print("h2b_vvov = ", np.linalg.norm(h2b_vvov.flatten()))
    print("h2b_vvvo = ", np.linalg.norm(h2b_vvvo.flatten()))
    print("h2c_vvov = ", np.linalg.norm(h2c_vvov.flatten()))
    print("")

    # h(oooo)
    h2a_oooo = np.zeros((noa, noa, noa, noa))
    h2b_oooo = np.zeros((noa, nob, noa, nob))
    h2c_oooo = np.zeros((nob, nob, nob, nob))
    for m in range(no):
        for n in range(no):
            for i in range(no):
                for j in range(no):
                    hmatel = H2[o, o, o, o][m, n, i, j]
                    if m % 2 == 0 and n % 2 == 0 and i % 2 == 0 and j % 2 == 0:
                        h2a_oooo[m // 2, n // 2, i // 2, j // 2] = hmatel
                    if m % 2 == 1 and n % 2 == 1 and i % 2 == 1 and j % 2 == 1:
                        h2c_oooo[(m - 1) // 2, (n - 1) // 2, (i - 1) // 2, (j - 1) // 2] = hmatel
                    if m % 2 == 0 and n % 2 == 1 and i % 2 == 0 and j % 2 == 1:
                        h2b_oooo[m // 2, (n - 1) // 2, i // 2, (j - 1) // 2] = hmatel
    print("h2a_oooo = ", np.linalg.norm(h2a_oooo.flatten()))
    print("h2b_oooo = ", np.linalg.norm(h2b_oooo.flatten()))
    print("h2c_oooo = ", np.linalg.norm(h2c_oooo.flatten()))
    print("")

    # h(vvvv)
    h2a_vvvv = np.zeros((nua, nua, nua, nua))
    h2b_vvvv = np.zeros((nua, nub, nua, nub))
    h2c_vvvv = np.zeros((nub, nub, nub, nub))
    for a in range(nu):
        for b in range(nu):
            for e in range(nu):
                for f in range(nu):
                    hmatel = H2[v, v, v, v][a, b, e, f]
                    if a % 2 == 0 and b % 2 == 0 and e % 2 == 0 and f % 2 == 0:
                        h2a_vvvv[a // 2, b // 2, e // 2, f // 2] = hmatel
                    if a % 2 == 1 and b % 2 == 1 and e % 2 == 1 and f % 2 == 1:
                        h2c_vvvv[(a - 1) // 2, (b - 1) // 2, (e - 1) // 2, (f - 1) // 2] = hmatel
                    if a % 2 == 0 and b % 2 == 1 and e % 2 == 0 and f % 2 == 1:
                        h2b_vvvv[a // 2, (b - 1) // 2, e // 2, (f - 1) // 2] = hmatel
    print("h2a_vvvv = ", np.linalg.norm(h2a_vvvv.flatten()))
    print("h2b_vvvv = ", np.linalg.norm(h2b_vvvv.flatten()))
    print("h2c_vvvv = ", np.linalg.norm(h2c_vvvv.flatten()))
    print("")

    # h(voov)
    h2a_voov = np.zeros((nua, noa, noa, nua))
    h2b_voov = np.zeros((nua, nob, noa, nub))
    h2b_vovo = np.zeros((nua, nob, nua, nob))
    h2b_ovov = np.zeros((noa, nub, noa, nub))
    h2b_ovvo = np.zeros((noa, nub, nua, nob))
    h2c_voov = np.zeros((nub, nob, nob, nub))
    for a in range(nu):
        for m in range(no):
            for i in range(no):
                for e in range(nu):
                    hmatel = H2[v, o, o, v][a, m, i, e]
                    if a % 2 == 0 and m % 2 == 0 and i % 2 == 0 and e % 2 == 0:
                        h2a_voov[a // 2, m // 2, i // 2, e // 2] = hmatel
                    if a % 2 == 1 and m % 2 == 1 and i % 2 == 1 and e % 2 == 1:
                        h2c_voov[(a - 1) // 2, (m - 1) // 2, (i - 1) // 2, (e - 1) // 2] = hmatel
                    if a % 2 == 0 and m % 2 == 1 and i % 2 == 0 and e % 2 == 1:
                        h2b_voov[a // 2, (m - 1) // 2, i // 2, (e - 1) // 2] = hmatel
                    if a % 2 == 1 and m % 2 == 0 and i % 2 == 1 and e % 2 == 0:
                        h2b_ovvo[m // 2, (a - 1) // 2, e // 2, (i - 1) // 2] = hmatel
                    if a % 2 == 0 and m % 2 == 1 and i % 2 == 1 and e % 2 == 0:
                        h2b_vovo[a // 2, (m - 1) // 2, e // 2, (i - 1) // 2] = -hmatel # Minus sign, watch out!
                    if a % 2 == 1 and m % 2 == 0 and i % 2 == 0 and e % 2 == 1:
                        h2b_ovov[m // 2, (a - 1) // 2, i // 2, (e - 1) // 2] = -hmatel # Minus sign, watch out!

    print("h2a_voov = ", np.linalg.norm(h2a_voov.flatten()))
    print("h2b_voov = ", np.linalg.norm(h2b_voov.flatten()))
    print("h2b_ovvo = ", np.linalg.norm(h2b_ovvo.flatten()), "Max value = ", max(h2b_ovvo.flatten()))
    print("h2b_vovo = ", np.linalg.norm(h2b_vovo.flatten()), "Max value = ", max(h2b_vovo.flatten()))
    print("h2b_ovov = ", np.linalg.norm(h2b_ovov.flatten()), "Max value = ", max(h2b_ovov.flatten()))
    print("h2c_voov = ", np.linalg.norm(h2c_voov.flatten()))
    print("")

    # x(vooo)
    x2a_vooo = np.zeros((nua, noa, noa, noa))
    x2b_vooo = np.zeros((nua, nob, noa, nob))
    x2b_ovoo = np.zeros((noa, nub, noa, nob))
    x2c_vooo = np.zeros((nub, nob, nob, nob))
    for a in range(nu):
        for m in range(no):
            for i in range(no):
                for j in range(no):
                    hmatel = x_vooo[a, m, i, j]
                    if a % 2 == 0 and m % 2 == 0 and i % 2 == 0 and j % 2 == 0:
                        x2a_vooo[a // 2, m // 2, i // 2, j // 2] = hmatel
                    if a % 2 == 1 and m % 2 == 1 and i % 2 == 1 and j % 2 == 1:
                        x2c_vooo[(a - 1) // 2, (m - 1) // 2, (i - 1) // 2, (j - 1) // 2] = hmatel
                    if a % 2 == 0 and m % 2 == 1 and i % 2 == 0 and j % 2 == 1:
                        x2b_vooo[a // 2, (m - 1) // 2, i // 2, (j - 1) // 2] = hmatel
                    if a % 2 == 1 and m % 2 == 0 and i % 2 == 1 and j % 2 == 0:
                        x2b_ovoo[m // 2, (a - 1) // 2, j // 2, (i - 1) // 2] = hmatel
    print("x2a_vooo = ", np.linalg.norm(x2a_vooo.flatten()))
    print("x2b_vooo = ", np.linalg.norm(x2b_vooo.flatten()))
    print("x2b_ovoo = ", np.linalg.norm(x2b_ovoo.flatten()))
    print("x2c_vooo = ", np.linalg.norm(x2c_vooo.flatten()))
    print("")

    # x(vvov)
    x2a_vvov = np.zeros((nua, nua, noa, nua))
    x2b_vvov = np.zeros((nua, nub, noa, nub))
    x2b_vvvo = np.zeros((nua, nub, nua, nob))
    x2c_vvov = np.zeros((nub, nub, nob, nub))
    for a in range(nu):
        for b in range(nu):
            for i in range(no):
                for e in range(nu):
                    hmatel = x_vvov[a, b, i, e]
                    if a % 2 == 0 and b % 2 == 0 and i % 2 == 0 and e % 2 == 0:
                        x2a_vvov[a // 2, b // 2, i // 2, e // 2] = hmatel
                    if a % 2 == 1 and b % 2 == 1 and i % 2 == 1 and e % 2 == 1:
                        x2c_vvov[(a - 1) // 2, (b - 1) // 2, (i - 1) // 2, (e - 1) // 2] = hmatel
                    if a % 2 == 0 and b % 2 == 1 and i % 2 == 0 and e % 2 == 1:
                        x2b_vvov[a // 2, (b - 1) // 2, i // 2, (e - 1) // 2] = hmatel
                    if a % 2 == 1 and b % 2 == 0 and i % 2 == 1 and e % 2 == 0:
                        x2b_vvvo[b // 2, (a - 1) // 2, e // 2, (i - 1) // 2] = hmatel
    print("x2a_vvov = ", np.linalg.norm(x2a_vvov.flatten()))
    print("x2b_vvov = ", np.linalg.norm(x2b_vvov.flatten()))
    print("x2b_vvvo = ", np.linalg.norm(x2b_vvvo.flatten()))
    print("x2c_vvov = ", np.linalg.norm(x2c_vvov.flatten()))
    print("")

    # x(oooo)
    x2a_oooo = np.zeros((noa, noa, noa, noa))
    x2b_oooo = np.zeros((noa, nob, noa, nob))
    x2c_oooo = np.zeros((nob, nob, nob, nob))
    for m in range(no):
        for n in range(no):
            for i in range(no):
                for j in range(no):
                    hmatel = x_oooo[m, n, i, j]
                    if m % 2 == 0 and n % 2 == 0 and i % 2 == 0 and j % 2 == 0:
                        x2a_oooo[m // 2, n // 2, i // 2, j // 2] = hmatel
                    if m % 2 == 1 and n % 2 == 1 and i % 2 == 1 and j % 2 == 1:
                        x2c_oooo[(m - 1) // 2, (n - 1) // 2, (i - 1) // 2, (j - 1) // 2] = hmatel
                    if m % 2 == 0 and n % 2 == 1 and i % 2 == 0 and j % 2 == 1:
                        x2b_oooo[m // 2, (n - 1) // 2, i // 2, (j - 1) // 2] = hmatel
    print("x2a_oooo = ", np.linalg.norm(x2a_oooo.flatten()))
    print("x2b_oooo = ", np.linalg.norm(x2b_oooo.flatten()))
    print("x2c_oooo = ", np.linalg.norm(x2c_oooo.flatten()))
    print("")

    # x(vvvv)
    x2a_vvvv = np.zeros((nua, nua, nua, nua))
    x2b_vvvv = np.zeros((nua, nub, nua, nub))
    x2c_vvvv = np.zeros((nub, nub, nub, nub))
    for a in range(nu):
        for b in range(nu):
            for e in range(nu):
                for f in range(nu):
                    hmatel = x_vvvv[a, b, e, f]
                    if a % 2 == 0 and b % 2 == 0 and e % 2 == 0 and f % 2 == 0:
                        x2a_vvvv[a // 2, b // 2, e // 2, f // 2] = hmatel
                    if a % 2 == 1 and b % 2 == 1 and e % 2 == 1 and f % 2 == 1:
                        x2c_vvvv[(a - 1) // 2, (b - 1) // 2, (e - 1) // 2, (f - 1) // 2] = hmatel
                    if a % 2 == 0 and b % 2 == 1 and e % 2 == 0 and f % 2 == 1:
                        x2b_vvvv[a // 2, (b - 1) // 2, e // 2, (f - 1) // 2] = hmatel
    print("x2a_vvvv = ", np.linalg.norm(x2a_vvvv.flatten()))
    print("x2b_vvvv = ", np.linalg.norm(x2b_vvvv.flatten()))
    print("x2c_vvvv = ", np.linalg.norm(x2c_vvvv.flatten()))
    print("")

    # x(voov)
    x2a_voov = np.zeros((nua, noa, noa, nua))
    x2b_voov = np.zeros((nua, nob, noa, nub))
    x2b_vovo = np.zeros((nua, nob, nua, nob))
    x2b_ovov = np.zeros((noa, nub, noa, nub))
    x2b_ovvo = np.zeros((noa, nub, nua, nob))
    x2c_voov = np.zeros((nub, nob, nob, nub))
    for a in range(nu):
        for m in range(no):
            for i in range(no):
                for e in range(nu):
                    hmatel = x_voov[a, m, i, e]
                    if a % 2 == 0 and m % 2 == 0 and i % 2 == 0 and e % 2 == 0:
                        x2a_voov[a // 2, m // 2, i // 2, e // 2] = hmatel
                    if a % 2 == 1 and m % 2 == 1 and i % 2 == 1 and e % 2 == 1:
                        x2c_voov[(a - 1) // 2, (m - 1) // 2, (i - 1) // 2, (e - 1) // 2] = hmatel
                    if a % 2 == 0 and m % 2 == 1 and i % 2 == 0 and e % 2 == 1:
                        x2b_voov[a // 2, (m - 1) // 2, i // 2, (e - 1) // 2] = hmatel
                    if a % 2 == 1 and m % 2 == 0 and i % 2 == 1 and e % 2 == 0:
                        x2b_ovvo[m // 2, (a - 1) // 2, e // 2, (i - 1) // 2] = hmatel
                    if a % 2 == 0 and m % 2 == 1 and i % 2 == 1 and e % 2 == 0:
                        x2b_vovo[a // 2, (m - 1) // 2, e // 2, (i - 1) // 2] = -hmatel # Minus sign, watch out!
                    if a % 2 == 1 and m % 2 == 0 and i % 2 == 0 and e % 2 == 1:
                        x2b_ovov[m // 2, (a - 1) // 2, i // 2, (e - 1) // 2] = -hmatel # Minus sign, watch out!

    print("x2a_voov = ", np.linalg.norm(x2a_voov.flatten()))
    print("x2b_voov = ", np.linalg.norm(x2b_voov.flatten()))
    print("x2b_ovvo = ", np.linalg.norm(x2b_ovvo.flatten()), "Max value = ", max(x2b_ovvo.flatten()))
    print("x2b_vovo = ", np.linalg.norm(x2b_vovo.flatten()), "Max value = ", max(x2b_vovo.flatten()))
    print("x2b_ovov = ", np.linalg.norm(x2b_ovov.flatten()), "Max value = ", max(x2b_ovov.flatten()))
    print("x2c_voov = ", np.linalg.norm(x2c_voov.flatten()))


    #print("error in vooo = ", np.linalg.norm(I_vooo.flatten() - h_vooo.flatten()))
    #print("error in vvov = ", np.linalg.norm(I_vvov.flatten() - h_vvov.flatten()))
    #print("error in oooo = ", np.linalg.norm(I_oooo.flatten() - h_oooo.flatten()))
    #print("error in vvvv = ", np.linalg.norm(I_vvvv.flatten() - h_vvvv.flatten()))


def compute_cc3_intermediates(g, t1, o, v):

    # h(vvov) and h(vooo) intermediates resulting from exp(-T1) H_N exp(T1)
    Q1 = -np.einsum("mnfe,an->amef", g[o, o, v, v], t1, optimize=True)
    I_vovv = g[v, o, v, v] + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)
    I_ooov = g[o, o, o, v] + 0.5 * Q1

    Q1 = -np.einsum("bmfe,am->abef", I_vovv, t1, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    I_vvvv = g[v, v, v, v] + Q1

    Q1 = +np.einsum("nmje,ei->mnij", I_ooov, t1, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    I_oooo = g[o, o, o, o] + Q1

    Q1 = g[v, o, o, v] + 0.5 * np.einsum("amef,ei->amif", g[v, o, v, v], t1, optimize=True)
    Q1 = np.einsum("amif,fj->amij", Q1, t1, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    I_vooo = g[v, o, o, o] + Q1 - np.einsum("nmij,an->amij", I_oooo, t1, optimize=True)

    Q1 = g[o, v, o, v] - 0.5 * np.einsum("mnie,bn->mbie", g[o, o, o, v], t1, optimize=True)
    Q1 = -np.einsum("mbie,am->abie", Q1, t1, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    I_vvov = g[v, v, o, v] + Q1 + np.einsum("abfe,fi->abie", I_vvvv, t1, optimize=True)

    return I_vooo, I_vvov, I_oooo, I_vvvv

def compute_eomcc3_intermediates(g, t1, o, v):

    # CCS Hbar elements
    h_voov = g[v, o, o, v] + (
            -np.einsum("nmie,an->amie", g[o, o, o, v], t1, optimize=True)
            +np.einsum("amfe,fi->amie", g[v, o, v, v], t1, optimize=True)
            -np.einsum("mnef,an,fi->amie", g[o, o, v, v], t1, t1, optimize=True)
    )
    h_oooo = 0.5 * g[o, o, o, o] + (
            +np.einsum("nmjf,fi->mnij", g[o, o, o, v], t1, optimize=True)
            +0.5 * np.einsum("mnef,ei,fj->mnij", g[o, o, v, v], t1, t1, optimize=True)
    )
    h_oooo -= np.transpose(h_oooo, (0, 1, 3, 2))
    h_vvvv = 0.5 * g[v, v, v, v] + (
            -np.einsum("anef,bn->abef", g[v, o, v, v], t1, optimize=True)
            +0.5 * np.einsum("mnef,am,bn->abef", g[o, o, v, v,], t1, t1, optimize=True)
    )
    h_vvvv -= np.transpose(h_vvvv, (1, 0, 2, 3))
    h_vooo = 0.5 * g[v, o, o, o] + (
            +np.einsum("amie,ej->amij", g[v, o, o, v], t1, optimize=True)
            -0.5 * np.einsum("nmij,an->amij", g[o, o, o, o], t1, optimize=True)
            -np.einsum("mnjf,an,fi->amij", g[o, o, o, v], t1, t1, optimize=True)
            +0.5 * np.einsum("amef,ei,fj->amij", g[v, o, v, v], t1, t1, optimize=True)
            -0.5 * np.einsum("mnef,an,fi,ej->amij", g[o, o, v, v], t1, t1, t1, optimize=True)
    )
    h_vooo -= np.transpose(h_vooo, (0, 1, 3, 2))
    h_vvov = 0.5 * g[v, v, o, v] + (
            -np.einsum("anie,bn->abie", g[v, o, o, v], t1, optimize=True)
            +0.5 * np.einsum("abfe,fi->abie", g[v, v, v, v], t1, optimize=True)
            -np.einsum("bnef,an,fi->abie", g[v, o, v, v], t1, t1, optimize=True)
            +0.5 * np.einsum("mnie,bn,am->abie", g[o, o, o, v], t1, t1, optimize=True)
            +0.5 * np.einsum("mnef,bm,an,fi->abie", g[o, o, v, v], t1, t1, t1, optimize=True)
    )
    h_vvov -= np.transpose(h_vvov, (1, 0, 2, 3))

    return h_vooo, h_vvov, h_oooo, h_vvvv, h_voov

def compute_t3(t2, I_vooo, I_vvov, e_abcijk):
    """Compute the projection of the CCSDT Hamiltonian on triples
        X[a, b, c, i, j, k] = < ijkabc | (H_N exp(T1+T2+T3))_C | 0 >
    """

    # h(vvov) and h(vooo) intermediates resulting from exp(-T1) H_N exp(T1)
    triples_res = -0.25 * np.einsum("amij,bcmk->abcijk", I_vooo, t2, optimize=True)
    triples_res += 0.25 * np.einsum("abie,ecjk->abcijk", I_vvov, t2, optimize=True)

    triples_res -= np.transpose(triples_res, (0, 1, 2, 3, 5, 4)) # (jk)
    triples_res -= np.transpose(triples_res, (0, 1, 2, 4, 3, 5)) + np.transpose(triples_res, (0, 1, 2, 5, 4, 3)) # (i/jk)
    triples_res -= np.transpose(triples_res, (0, 2, 1, 3, 4, 5)) # (bc)
    triples_res -= np.transpose(triples_res, (2, 1, 0, 3, 4, 5)) + np.transpose(triples_res, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return triples_res * e_abcijk

if __name__ == "__main__":
    main()

