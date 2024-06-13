import numpy as np

def compute_l3(l1, l2, H1, H2, o, v):
    """Compute < 0 | ( 1 + L1 + L2 ) * H(2) | ijkabc >"""
    l3 = 0.25 * np.einsum("aeij,ekbc->abcijk", l2, H2[v, o, v, v], optimize=True)
    l3 -= 0.25 * np.einsum("abim,jkmc->abcijk", l2, H2[o, o, o, v], optimize=True)
    l3 += 0.25 * np.einsum("ai,jkbc->abcijk", l1, H2[o, o, v, v], optimize=True)
    l3 += 0.25 * np.einsum("abij,kc->abcijk", l2, H1[o, v], optimize=True)
    # antisymmetrize
    l3 -= np.transpose(l3, (1, 0, 2, 3, 4, 5)) + np.transpose(l3, (2, 1, 0, 3, 4, 5)) # (a/bc)
    l3 -= np.transpose(l3, (0, 2, 1, 3, 4, 5)) # (bc)
    l3 -= np.transpose(l3, (0, 1, 2, 4, 3, 5)) + np.transpose(l3, (0, 1, 2, 5, 4, 3)) # (i/jk)
    l3 -= np.transpose(l3, (0, 1, 2, 3, 5, 4)) # (jk)
    return l3

def compute_m3(t1, t2, H1, H2, o, v):
    """Compute < ijkabc | H(2) | 0 >"""
    I_vooo = H2[v, o, o, o] - np.einsum("me,aeij->amij", H1[o, v], t2, optimize=True)
    m3 = -0.25 * np.einsum("amij,bcmk->abcijk", I_vooo, t2, optimize=True)
    m3 += 0.25 * np.einsum("abie,ecjk->abcijk", H2[v, v, o, v], t2, optimize=True) 
    # antisymmetrize
    m3 -= np.transpose(m3, (1, 0, 2, 3, 4, 5)) + np.transpose(m3, (2, 1, 0, 3, 4, 5)) # (a/bc)
    m3 -= np.transpose(m3, (0, 2, 1, 3, 4, 5)) # (bc)
    m3 -= np.transpose(m3, (0, 1, 2, 4, 3, 5)) + np.transpose(m3, (0, 1, 2, 5, 4, 3)) # (i/jk)
    m3 -= np.transpose(m3, (0, 1, 2, 3, 5, 4)) # (jk)
    return m3

def compute_m4(t1, t2, H1, H2, o, v):
    # <ijklabcd | H(2) | 0 >
    m4 = -(144.0 / 576.0) * np.einsum("amie,bcmk,edjl->abcdijkl", H2[v, o, o, v], t2, t2, optimize=True)  # (jl/i/k)(bc/a/d) = 12 * 12 = 144
    m4 += (36.0 / 576.0) * np.einsum("mnij,adml,bcnk->abcdijkl", H2[o, o, o, o], t2, t2, optimize=True)   # (ij/kl)(bc/ad) = 6 * 6 = 36
    m4 += (36.0 / 576.0) * np.einsum("abef,fcjk,edil->abcdijkl", H2[v, v, v, v], t2, t2, optimize=True)   # (jk/il)(ab/cd) = 6 * 6 = 36
    # antisymmetrize
    m4 -= np.transpose(m4, (0, 1, 2, 3, 4, 6, 5, 7)) # (jk)
    m4 -= np.transpose(m4, (0, 1, 2, 3, 4, 7, 6, 5)) + np.transpose(m4, (0, 1, 2, 3, 4, 5, 7, 6)) # (l/jk)
    m4 -= np.transpose(m4, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(m4, (0, 1, 2, 3, 6, 5, 4, 7)) + np.transpose(m4, (0, 1, 2, 3, 7, 5, 6, 4)) # (i/jkl)
    m4 -= np.transpose(m4, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    m4 -= np.transpose(m4, (0, 3, 2, 1, 4, 5, 6, 7)) + np.transpose(m4, (0, 1, 3, 2, 4, 5, 6, 7)) # (d/bc)
    m4 -= np.transpose(m4, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(m4, (2, 1, 0, 3, 4, 5, 6, 7)) + np.transpose(m4, (3, 1, 2, 0, 4, 5, 6, 7)) # (a/bcd)
    return m4

def compute_l4(l1, l2, H1, H2, o, v):
    # < 0 | L2 * H(2) | ijklabcd >
    l4 = (36.0 / 576.0) * np.einsum("abij,cdkl->abcdijkl", l2, H2[o, o, v, v], optimize=True)
    # antisymmetrize
    l4 -= np.transpose(l4, (0, 1, 2, 3, 4, 6, 5, 7)) # (jk)
    l4 -= np.transpose(l4, (0, 1, 2, 3, 4, 7, 6, 5)) + np.transpose(l4, (0, 1, 2, 3, 4, 5, 7, 6)) # (l/jk)
    l4 -= np.transpose(l4, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(l4, (0, 1, 2, 3, 6, 5, 4, 7)) + np.transpose(l4, (0, 1, 2, 3, 7, 5, 6, 4)) # (i/jkl)
    l4 -= np.transpose(l4, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    l4 -= np.transpose(l4, (0, 3, 2, 1, 4, 5, 6, 7)) + np.transpose(l4, (0, 1, 3, 2, 4, 5, 6, 7)) # (d/bc)
    l4 -= np.transpose(l4, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(l4, (2, 1, 0, 3, 4, 5, 6, 7)) + np.transpose(l4, (3, 1, 2, 0, 4, 5, 6, 7)) # (a/bcd)
    return l4

def compute_i2(l3, m3):
    i2 = (1.0 / 36.0) * np.einsum("abcijk,abcijk->", l3, m3, optimize=True)
    return i2

def compute_i3(l3, t1, t2, m3, H1, H2, o, v, m4=None):

    I_vvov = H2[v, v, o, v] + (
              -0.5 * np.einsum("mnef,abfimn->abie", H2[o, o, v, v], m3, optimize=True)
              +np.einsum("me,abim->abie", H1[o, v], t2, optimize=True)
    )
    I_vooo = H2[v, o, o, o] + 0.5 * np.einsum("mnef,aefijn->amij", H2[o, o, v, v], m3, optimize=True)

    # CCSDT-like parts
    triples_res = -0.25 * np.einsum("amij,bcmk->abcijk", I_vooo, t2, optimize=True)
    triples_res += 0.25 * np.einsum("abie,ecjk->abcijk", I_vvov, t2, optimize=True)
    triples_res -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H1[o, o], m3, optimize=True)
    triples_res += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H1[v, v], m3, optimize=True)
    triples_res += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H2[o, o, o, o], m3, optimize=True)
    triples_res += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H2[v, v, v, v], m3, optimize=True)
    triples_res += 0.25 * np.einsum("cmke,abeijm->abcijk", H2[v, o, o, v], m3, optimize=True)
    # Parts with quadruples (c.f. T4 contributions in T3 projections in CCSDTQ)
    if m4 is not None:
        triples_res += (1.0 / 36.0) * np.einsum("me,abceijkm->abcijk", H1[o, v], m4, optimize=True) # (1) = 1
        triples_res += (1.0 / 24.0) * np.einsum("cnef,abefijkn->abcijk", H2[v, o, v, v], m4, optimize=True) # (c/ab) = 3
        triples_res -= (1.0 / 24.0) * np.einsum("mnkf,abcfijmn->abcijk", H2[o, o, o, v], m4, optimize=True) # (k/ij) = 3

    # antisymmetrize
    triples_res -= np.transpose(triples_res, (0, 1, 2, 3, 5, 4)) # (jk)
    triples_res -= np.transpose(triples_res, (0, 1, 2, 4, 3, 5)) + np.transpose(triples_res, (0, 1, 2, 5, 4, 3)) # (i/jk)
    triples_res -= np.transpose(triples_res, (0, 2, 1, 3, 4, 5)) # (bc)
    triples_res -= np.transpose(triples_res, (2, 1, 0, 3, 4, 5)) + np.transpose(triples_res, (1, 0, 2, 3, 4, 5)) # (a/bc)

    # triples part
    i3 = (1.0 / 36.0) * np.einsum("abcijk,abcijk->", l3, triples_res, optimize=True)

    return i3

def kernel(T, L, Ecorr, H1, H2, o, v):
    print("    ==> CCSD-CMX(2) energy correction <==\n")

    t1, t2 = T
    l1, l2 = L

    # compute L3 and M3
    print("     Computing 3-body moments and left vector...")
    m3 = compute_m3(t1, t2, H1, H2, o, v)
    l3 = compute_l3(l1, l2, H1, H2, o, v)
    # compute L4 and M4
    #print("     Computing 4-body moments and left vector...")
    #m4 = compute_m4(t1, t2, H1, H2, o, v)
    #l4 = compute_l4(l1, l2, H1, H2, o, v)
    m4 = None

    I2 = compute_i2(l3, m3)

    I3 = compute_i3(l3, t1, t2, m3, H1, H2, o, v, m4=m4)
    # adjust I3 by the "renormalization" terms, -2*I2*E0(CCSD) 
    I3 -= 2.0 * I2 * Ecorr

    print("\n     CCSD-CMX(2) Summary")
    print("     -----------------------")
    print(f"     I1 = {Ecorr}")
    print(f"     I2 = {I2}")
    print(f"     I3 = {I3}")
    #print(f"   I2^2 = {I2**2}")
    #print(f"   I2^2/I3 = {I2**2 / I3}")
    #print(f"   CMX(2) Correction to I1 = {-I2**2 / I3}") 
    print("")

    cmx2 = -I2**2 / I3

    return cmx2
