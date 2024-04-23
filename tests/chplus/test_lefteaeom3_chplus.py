import numpy as np
from miniccpy.driver import run_scf, run_cc_calc, run_guess, run_eomcc_calc, get_hbar, run_lefteomcc_calc

basis = '6-31g'
nfrozen = 0

# Define molecule geometry and basis set
geom = [['C', (0.0, 0.0, 2.1773/2)], 
        ['H', (0.0, 0.0, -2.1773/2)]]

fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, charge=1, unit="Bohr", symmetry="C2V")

T, Ecorr  = run_cc_calc(fock, g, o, v, method='ccsd')

H1, H2 = get_hbar(T, fock, g, o, v, method='ccsd')

R, omega_guess = run_guess(H1, H2, o, v, 10, method="eacis")
R, omega, r0 = run_eomcc_calc(R, omega_guess, T, H1, H2, o, v, method="eaeom3", state_index=[0], max_size=40)
L, omega_left = run_lefteomcc_calc(R, omega, T, H1, H2, o, v, method='left_eaeom3')

l1, l2, l3 = L[0]

print(np.linalg.norm(l3.flatten()))

nu, _, no = l2.shape
l3a = np.zeros((8, 8, 8, 3, 3))
l3b = np.zeros((8, 8, 8, 3, 3))
l3c = np.zeros((8, 8, 8, 3, 3))
for a in range(nu):
    for b in range(a + 1, nu):
        for c in range(b + 1, nu):
            for j in range(no):
                for k in range(j + 1, no):
                    val = l3[a, b, c, j, k]
                    #if abs(val) > 1.0e-09:
                    #    print(a, b, c, j, k)
                    if a % 2 == 0 and b % 2 == 0 and c % 2 == 0 and j % 2 == 0 and k % 2 == 0:
                        aa = a // 2
                        bb = b // 2
                        cc = c // 2
                        jj = j // 2
                        kk = k // 2
                        l3a[aa, bb, cc, jj, kk] = val
                        l3a[aa, cc, bb, jj, kk] = -val
                        l3a[bb, aa, cc, jj, kk] = -val
                        l3a[bb, cc, aa, jj, kk] = val
                        l3a[cc, aa, bb, jj, kk] = val
                        l3a[cc, bb, aa, jj, kk] = -val
                        l3a[aa, bb, cc, kk, jj] = -val
                        l3a[aa, cc, bb, kk, jj] = val
                        l3a[bb, aa, cc, kk, jj] = val
                        l3a[bb, cc, aa, kk, jj] = -val
                        l3a[cc, aa, bb, kk, jj] = -val
                        l3a[cc, bb, aa, kk, jj] = val
                    if a % 2 == 0 and b % 2 == 0 and c % 2 == 1 and j % 2 == 0 and k % 2 == 1:
                        aa = a // 2
                        bb = b // 2
                        cc = (c - 1) // 2
                        jj = j // 2
                        kk = (k - 1) // 2
                        l3b[aa, bb, cc, jj, kk] = val
                        l3b[bb, aa, cc, jj, kk] = -val
                    if a % 2 == 0 and b % 2 == 1 and c % 2 == 1 and j % 2 == 1 and k % 2 == 1:
                        aa = a // 2
                        bb = (b - 1) // 2
                        cc = (c - 1) // 2
                        jj = (j - 1) // 2
                        kk = (k - 1) // 2
                        l3c[aa, bb, cc, jj, kk] = val
                        l3c[aa, cc, bb, jj, kk] = -val
                        l3c[aa, bb, cc, kk, jj] = -val
                        l3c[aa, cc, bb, kk, jj] = val
                        
print("norm of l3a = ", np.linalg.norm(l3a.flatten()))
print("norm of l3b = ", np.linalg.norm(l3b.flatten()))
print("norm of l3c = ", np.linalg.norm(l3c.flatten()))



