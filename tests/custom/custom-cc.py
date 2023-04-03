import numpy as np
from miniccpy.driver import run_cc_calc

nelec = 2
norb = 4

h1 = np.random.rand(norb, norb)
h1 = 0.5*(h1 + h1.T)

for i in range(norb):
    h1[i, i] = -(norb - i) * 10

for i in range(nelec):
    for a in range(nelec, norb):
        h1[a, i] = 0.0
        h1[i, a] = 0.0

print(h1)

h2 = 0.01 * np.random.rand(norb, norb, norb, norb)
h2 -= np.einsum("pqrs->pqsr", h2, optimize=True)

o = slice(0, nelec)
v = slice(nelec, norb)

T, E_corr = run_cc_calc(h1, h2, o, v, method='ccsd')







