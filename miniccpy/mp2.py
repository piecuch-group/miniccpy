import numpy as np

def kernel(fock, g, o, v):
    print("    ==> MP2 energy correction <==")
    nu, no = fock[v, o].shape
    energy = 0.0
    for a in range(nu):
        for b in range(a + 1, nu):
            for i in range(no):
                for j in range(i + 1, no):
                    denom = fock[o, o][i, i] + fock[o, o][j, j] - fock[v, v][a, a] - fock[v, v][b, b]
                    energy += g[o, o, v, v][i, j, a, b] * g[v, v, o, o][a, b, i, j] / denom
    return energy
