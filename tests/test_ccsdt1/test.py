import numpy as np
from miniccpy.driver import run_scf, run_cc_calc

from miniccpy.hbar import get_rccs_intermediates, get_rccsd_intermediates

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
    #triples_res += 0.5 * np.einsum("mnij,abcmnk->abcijk", H2[o, o, o, o], t3, optimize=True)
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
    return triples_res

def triples_res_111111(t1, t2, t3, f, g, o, v, P, p, H, h):
    # ABCIJK [(AI)(BJ)(CK)] -> symmetric wrt [1 + P(AI/BJ)][1 + P(AI/CK) + P(BJ/CK)]
    return

def triples_res_110111(t1, t2, t3, f, g, o, v, P, p, H, h):
    # ABcIJK [(AI)(BJ)(cK)]-> symmetric wrt P(AI/BJ)
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
    d1 =  np.einsum("AE,EBcIJK->ABcIJK", H1[v, v][P, P], t3[P, P, p, H, H, H], optimize=True)
    d1 += np.einsum("Ae,BecJIK->ABcIJK", H1[v, v][P, p], t3[P, p, p, H, H, H], optimize=True)
    d1 += 0.5 * np.einsum("cE,ABEIJK->ABcIJK", H1[v, v][p, P], t3[P, P, P, H, H, H], optimize=True)
    d1 += 0.5 * np.einsum("ce,ABeIJK->ABcIJK", H1[v, v][p, p], t3[P, P, p, H, H, H], optimize=True)
    d1 += d1.transpose(1, 0, 2, 4, 3, 5) # P(ai/bj)

    d2 = -np.einsum("MI,ABcMJK->ABcIJK", H1[o, o][H, H], t3[P, P, p, H, H, H], optimize=True)
    d2 -= np.einsum("mI,ABcmJK->ABcIJK", H1[o, o][h, H], t3[P, P, p, h, H, H], optimize=True)
    d2 -= 0.5 * np.einsum("MK,ABcIJM->ABcIJK", H1[o, o][H, H], t3[P, P, p, H, H, H], optimize=True)
    d2 -= 0.5 * np.einsum("mK,AcBImJ->ABcIJK", H1[o, o][h, H], t3[P, p, P, H, h, H], optimize=True)
    d2 += d2.transpose(1, 0, 2, 4, 3, 5) # P(ai/bj)

    # Issues with d3 and d4? Error is small but increases by orders of magnitude..
    d3 = 0.5 * np.einsum("MNIJ,ABcMNK->ABcIJK", H2[o, o, o, o][H, H, H, H], t3[P, P, p, H, H, H], optimize=True)
    d3 += np.einsum("mNIJ,ABcmNK->ABcIJK", H2[o, o, o, o][h, H, H, H], t3[P, P, p, h, H, H], optimize=True)
    d3 += 0.5 * np.einsum("mnIJ,ABcmnK->ABcIJK", H2[o, o, o, o][h, h, H, H], t3[P, P, p, h, h, H], optimize=True)
    d3 += np.einsum("MNJK,ABcIMN->ABcIJK", H2[o, o, o, o][H, H, H, H], t3[P, P, p, H, H, H], optimize=True)
    d3 += np.einsum("MnJK,AcBInM->ABcIJK", H2[o, o, o, o][H, h, H, H], t3[P, p, P, H, h, H], optimize=True)
    d3 += np.einsum("mNJK,ABcImN->ABcIJK", H2[o, o, o, o][h, H, H, H], t3[P, P, p, H, h, H], optimize=True)
    d3 += np.einsum("mnJK,BcAmnI->ABcIJK", H2[o, o, o, o][h, h, H, H], t3[P, p, P, h, h, H], optimize=True)
    d3 += d3.transpose(1, 0, 2, 4, 3, 5) # P(ai/bj)

    d4 = 0.5 * np.einsum("ABEF,EFcIJK->ABcIJK", H2[v, v, v, v][P, P, P, P], t3[P, P, p, H, H, H], optimize=True)
    d4 += np.einsum("ABEf,EfcIJK->ABcIJK", H2[v, v, v, v][P, P, P, p], t3[P, p, p, H, H, H], optimize=True)
    d4 += np.einsum("AcEF,EBFIJK->ABcIJK", H2[v, v, v, v][P, p, P, P], t3[P, P, P, H, H, H], optimize=True)
    d4 += np.einsum("AceF,eBFIJK->ABcIJK", H2[v, v, v, v][P, p, p, P], t3[p, P, P, H, H, H], optimize=True)
    d4 += np.einsum("AcEf,EBfIJK->ABcIJK", H2[v, v, v, v][P, p, P, p], t3[P, P, p, H, H, H], optimize=True)
    d4 += np.einsum("Acef,eBfIJK->ABcIJK", H2[v, v, v, v][P, p, p, p], t3[p, P, p, H, H, H], optimize=True)
    d4 += d4.transpose(1, 0, 2, 4, 3, 5)

    # need to break up the spin summation into its constituents unfortunately...
    #d5 = np.einsum("AMIE,EBcMJK->ABcIJK", H2[v, o, o, v][P, H, H, P], 2.0 * t3[P, P, p, H, H, H], optimize=True)
    #d5 += np.einsum("AmIE,EBcmJK->ABcIJK", H2[v, o, o, v][P, h, H, P], 2.0 * t3[P, P, p, h, H, H], optimize=True)
    #d5 += np.einsum("AMIe,eBcMJK->ABcIJK", H2[v, o, o, v][P, H, H, p], 2.0 * t3[p, P, p, H, H, H], optimize=True)
    #d5 += np.einsum("AmIe,eBcmJK->ABcIJK", H2[v, o, o, v][P, h, H, p], 2.0 * t3[p, P, p, h, H, H], optimize=True)
    #d5 += 0.5 * np.einsum("cMKE,ABEIJM->ABcIJK", H2[v, o, o, v][p, H, H, P], 2.0 * t3[P, P, P, H, H, H], optimize=True)
    #d5 += 0.5 * np.einsum("cmKE,ABEIJm->ABcIJK", H2[v, o, o, v][p, h, H, P], 2.0 * t3[P, P, P, H, H, h], optimize=True)
    #d5 += 0.5 * np.einsum("cMKe,ABeIJM->ABcIJK", H2[v, o, o, v][p, H, H, p], 2.0 * t3[P, P, p, H, H, H], optimize=True)
    #d5 += 0.5 * np.einsum("cmKe,ABeIJm->ABcIJK", H2[v, o, o, v][p, h, H, p], 2.0 * t3[P, P, p, H, H, h], optimize=True)
    #d5 += d5.transpose(1, 0, 2, 4, 3, 5)
    # Can use t3_s as long as the spin-summed vertex (e,m) is consistently placed. Given that the full active-space
    # t3 array is of the t3(AbcijK), where b,c,i,j are full orbital indices, consider spin-summing b and j to form
    # t3_s and move the (e,m) contraction to the (b,j) position in all contractions.
    d5 = np.einsum("AMIE,EBcMJK->ABcIJK", H2[v, o, o, v][P, H, H, P], t3_s[P, P, p, H, H, H], optimize=True)
    d5 += np.einsum("AmIE,EBcmJK->ABcIJK", H2[v, o, o, v][P, h, H, P], t3_s[P, P, p, h, H, H], optimize=True)
    d5 += np.einsum("AMIe,eBcMJK->ABcIJK", H2[v, o, o, v][P, H, H, p], t3_s[p, P, p, H, H, H], optimize=True)
    d5 += np.einsum("AmIe,eBcmJK->ABcIJK", H2[v, o, o, v][P, h, H, p], t3_s[p, P, p, h, H, H], optimize=True)
    d5 += 0.5 * np.einsum("cMKE,EABMIJ->ABcIJK", H2[v, o, o, v][p, H, H, P], t3_s[P, P, P, H, H, H], optimize=True)
    d5 += 0.5 * np.einsum("cmKE,EABmIJ->ABcIJK", H2[v, o, o, v][p, h, H, P], t3_s[P, P, P, h, H, H], optimize=True)
    d5 += 0.5 * np.einsum("cMKe,eABMIJ->ABcIJK", H2[v, o, o, v][p, H, H, p], t3_s[p, P, P, H, H, H], optimize=True)
    d5 += 0.5 * np.einsum("cmKe,eABmIJ->ABcIJK", H2[v, o, o, v][p, h, H, p], t3_s[p, P, P, h, H, H], optimize=True)
    d5 += d5.transpose(1, 0, 2, 4, 3, 5)

    #triples_res += 0.5 * np.einsum("amie,ebcmjk->abcijk", H2[v, o, o, v], t3_s, optimize=True)
    #triples_res -= 0.25 * np.einsum("amei,ebcmjk->abcijk", H2[v, o, v, o], t3_s, optimize=True)
    #triples_res -= 0.5 * np.einsum("amei,ebcjmk->abcijk", H2[v, o, v, o], t3, optimize=True)
    #triples_res -= np.einsum("bmei,eacjmk->abcijk", H2[v, o, v, o], t3, optimize=True)
    #triples_res += triples_res.transpose(1, 0, 2, 4, 3, 5) # P(AI/BJ)
    #triples_res = d1 + d2# + d3# + d4# + d5
    triples_res = d1 + d2 + d5 + d4
    return triples_res

def triples_res_111011(t1, t2, t3, f, g, o, v, P, p, H, h):
    # ABCiJK [(Ai)(BJ)(CK)] -> symmetric wrt P(BJ/CK)
    return

def triples_res_110011(t1, t2, t3, f, g, o, v, P, p, H, h):
    # ABciJK [(Ai)(BJ)(cK)] -> no symmetry
    return

def triples_res_100111(t1, t2, t3, f, g, o, v, P, p, H, h):
    # AbcIJK [(AI)(bJ)(cK)] -> symmetric wrt P(bJ/cK)
    return

def triples_res_100011(t1, t2, t3, f, g, o, v, P, p, H, h):
    # AbciJK [(Ai)(bJ)(cK)] -> symmetric wrt P(bJ/cK)
    return

def triples_res_111001(t1, t2, t3, f, g, o, v, P, p, H, h):
    # ABCijK [(Ai)(Bj)(CK)] -> symmetric wrt P(Ai/Bj)
    return

def triples_res_110001(t1, t2, t3, f, g, o, v, P, p, H, h):
    # ABcijK [(Ai)(Bj)(cK)] -> symmetric wrt P(Ai/Bj)
    return

def triples_res_100001(t1, t2, t3, f, g, o, v, P, p, H, h):
    # AbcijK [(Ai)(bj)(cK)] -> no symmetry
    return

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
    t3 = zero_outside_active_space(t3, nacto, nactu)
    x3 = exact_triples_res(t1, t2, t3, fock, g, o, v)

    x3_110111 = triples_res_110111(t1, t2, t3, fock, g, o, v, P, p, H, h) 
    print("Error in 110111 = ", np.linalg.norm(x3[P, P, p, H, H, H] - x3_110111))




