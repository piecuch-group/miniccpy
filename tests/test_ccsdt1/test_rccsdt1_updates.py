import numpy as np
import time
from miniccpy.driver import run_scf, run_cc_calc

from miniccpy.hbar import get_rccs_intermediates, get_rccsd_intermediates
from miniccpy.rccsdt1 import *

def zero_outside_active_space(t3, nacto, nactu):

    nu, _, _, no, _, _ = t3.shape
    for a in range(nu):
        for b in range(a, nu):
            for c in range(b, nu):
                for i in range(no):
                    for j in range(i, no):
                        for k in range(j, no):
                            n_virt = 0
                            n_core = 0
                            if a >= nactu: n_virt += 1
                            if b >= nactu: n_virt += 1
                            if c >= nactu: n_virt += 1
                            if i < nacto: n_core += 1
                            if j < nacto: n_core += 1
                            if k < nacto: n_core += 1
                            #if n_core >= 3 or n_virt >= 3:
                            #    t3[a, b, c, i, j, k] = 0.0
                            #    t3[a, c, b, i, k, j] = 0.0 # P(bj/ck)
                            #    t3[b, a, c, j, i, k] = 0.0 # P(ai/bj)
                            #    t3[b, c, a, j, k, i] = 0.0 # P(ai/bj)P(bj/ck)
                            #    t3[c, b, a, k, j, i] = 0.0 # P(ai/ck)
                            #    t3[c, a, b, k, i, j] = 0.0 # P(ai/bj)P(ai/ck)
                            if n_core >= 3:
                                t3[:, :, :, i, j, k] *= 0.0
                                t3[:, :, :, i, k, j] *= 0.0 # P(bj/ck)
                                t3[:, :, :, j, i, k] *= 0.0 # P(ai/bj)
                                t3[:, :, :, j, k, i] *= 0.0 # P(ai/bj)P(bj/ck)
                                t3[:, :, :, k, j, i] *= 0.0 # P(ai/ck)
                                t3[:, :, :, k, i, j] *= 0.0 # P(ai/bj)P(ai/ck)
                            if n_virt >= 3:
                                t3[a, b, c, :, :, :] *= 0.0
                                t3[a, c, b, :, :, :] *= 0.0 # P(bj/ck)
                                t3[b, a, c, :, :, :] *= 0.0 # P(ai/bj)
                                t3[b, c, a, :, :, :] *= 0.0 # P(ai/bj)P(bj/ck)
                                t3[c, b, a, :, :, :] *= 0.0 # P(ai/ck)
                                t3[c, a, b, :, :, :] *= 0.0 # P(ai/bj)P(ai/ck)
    return t3

def exact_triples_res(t1, t2, t3, f, g, o, v):
    # partially spin-summed t3(AbcIjk) = 2*t3(abcijk) - t3(abcjik) - t3(abckji)
    t3_s = (
            2.0 * t3
            - t3.transpose(0, 1, 2, 4, 3, 5)
            - t3.transpose(0, 1, 2, 5, 4, 3)
    )
    # intermediates
    H1, H2 = get_rccsd_intermediates(t1, t2, f, g, o, v)
    I_vvov = H2[v, v, o, v] - np.einsum("nmfe,fabnim->abie", g[o, o, v, v], t3_s, optimize=True)
    I_vooo = H2[v, o, o, o] + (
            - np.einsum("me,aeik->amik", H1[o, v], t2, optimize=True)
            + np.einsum("nmfe,faenij->amij", g[o, o, v, v], t3_s, optimize=True)
    )

    # MM(2,3)B
    #triples_res = -np.einsum("amij,bcmk->abcijk", I_vooo, t2, optimize=True)
    #triples_res += np.einsum("abie,ecjk->abcijk", I_vvov, t2, optimize=True)
    # (HBar*T3)_C
    triples_res = 0.5 * np.einsum("ae,ebcijk->abcijk", H1[v, v], t3, optimize=True)
    triples_res -= 0.5 * np.einsum("mi,abcmjk->abcijk", H1[o, o], t3, optimize=True)
    triples_res += 0.5 * np.einsum("mnij,abcmnk->abcijk", H2[o, o, o, o], t3, optimize=True)
    triples_res += 0.5 * np.einsum("abef,efcijk->abcijk", H2[v, v, v, v], t3, optimize=True)
    triples_res += 0.5 * np.einsum("amie,ebcmjk->abcijk", H2[v, o, o, v], t3_s, optimize=True)
    # expanding out spin-summed vertex
    #triples_res += 0.5 * np.einsum("amie,ebcmjk->abcijk", H2[v, o, o, v], 2.0 * t3, optimize=True)
    #triples_res -= 0.5 * np.einsum("amie,ebcjmk->abcijk", H2[v, o, o, v], t3, optimize=True)
    #triples_res -= 0.5 * np.einsum("amie,ebckjm->abcijk", H2[v, o, o, v], t3, optimize=True)
    #
    #triples_res -= 0.25 * np.einsum("amei,ebcmjk->abcijk", H2[v, o, v, o], t3_s, optimize=True)
    #triples_res -= 0.5 * np.einsum("amei,ebcjmk->abcijk", H2[v, o, v, o], t3, optimize=True)
    #triples_res -= np.einsum("bmei,eacjmk->abcijk", H2[v, o, v, o], t3, optimize=True)
    # [1 + P(ai/bj)][1 + P(ai/ck) + P(bj/ck)] = 1 + P(ai/bj) + P(ai/ck) + P(bj/ck) + P(ai/bj)P(ai/ck) + P(ai/bj)P(bj/ck)
    triples_res += (    triples_res.transpose(1, 0, 2, 4, 3, 5)   # (ij)(ab)
                      + triples_res.transpose(2, 1, 0, 5, 4, 3)   # (ac)(ik)
                      + triples_res.transpose(0, 2, 1, 3, 5, 4)   # (bc)(jk)
                      + triples_res.transpose(2, 0, 1, 5, 3, 4)   # (ab)(ij)(ac)(ik)
                      + triples_res.transpose(1, 2, 0, 4, 5, 3) ) # (ab)(ij)(bc)(jk)
    nu, no = t1.shape
    for i in range(no):
        triples_res[:, :, :, i, i, i] *= 0.0
    for a in range(nu):
        triples_res[a, a, a, :, :, :] *= 0.0
    return triples_res

if __name__ == "__main__":
    basis = 'aug-cc-pvdz'
    nfrozen = 0

    nacto = 3
    nactu = 4

    geom = [['H', (0.0, 0.0, -1.0)], 
            ['F', (0.0, 0.0,  1.0)]]

    fock, g, e_hf, o, v = run_scf(geom, basis, nfrozen, maxit=200, unit="Angstrom", rhf=True)

    no, nu = fock[o, v].shape
    H = slice(no - nacto, no)
    h = slice(0, no - nacto)
    P = slice(0, nactu)
    p = slice(nactu, nu)

    print(fock[o, v][H, P].shape)
    print(fock[o, v][h, p].shape)

    T, E_corr = run_cc_calc(fock, g, o, v, method='rccsdt', maxit=80)

    #assert np.allclose(-0.221558736943, E_corr, atol=1.0e-08)

    t1, t2, t3 = T
    tic = time.time()
    t3 = zero_outside_active_space(t3, nacto, nactu)
    print("Space zeroing took", time.time() - tic, "s")
    tic = time.time()
    x3 = exact_triples_res(t1, t2, t3, fock, g, o, v)
    print("Exact update took", time.time() - tic, "s")
    tic = time.time()
    x3_110111 = triples_res_110111(t1, t2, t3, fock, g, o, v, P, p, H, h) 
    print("110111 update took", time.time() - tic, "s")
    print("Error in 110111 = ", np.linalg.norm(x3[P, P, p, H, H, H] - x3_110111))




