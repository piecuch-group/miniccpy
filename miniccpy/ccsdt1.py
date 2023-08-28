import time
import numpy as np
from miniccpy.energy import cc_energy, hf_energy, hf_energy_from_fock
from miniccpy.hbar import get_ccs_intermediates, get_ccsd_intermediates
from miniccpy.diis import DIIS

def active_slices(o, v, nacto, nactu):

    nfroz = o.start
    nelectrons = o.stop
    norbitals = v.stop

    nocc = nelectrons - nfroz
    nunocc = norbitals - nelectrons

    # Active-space slicing arrays for integrals (lives in full orbital space)
    H_ints = slice(nelectrons - nacto, nelectrons)
    h_ints = slice(nfroz, nelectrons - nacto)
    P_ints = slice(0, nactu)
    p_ints = slice(nactu, norbitals)

    # Active-space slicing arrays for cluster operator (lives in correlated orbital space)
    H = slice(nocc - nacto, nocc)
    h = slice(0, nocc - nacto)
    P = slice(0, nactu)
    p = slice(nactu, nunocc)

    ints_slice = {"H" : H_ints, "h" : h_ints, "P" : P_ints, "p" : p_ints}
    corr_slice = {"H" : H, "h" : h, "P" : P, "p" : p}

    return corr_slice, ints_slice

def singles_residual(t1, t2, t3, f, g, o, v, corr_slice, ints_slice):
    """Compute the projection of the CCSDT Hamiltonian on singles
        X[a, i] = < ia | (H_N exp(T1+T2+T3))_C | 0 >
    """

    chi_vv = f[v, v] + np.einsum("anef,fn->ae", g[v, o, v, v], t1, optimize=True)

    chi_oo = f[o, o] + np.einsum("mnif,fn->mi", g[o, o, o, v], t1, optimize=True)

    h_ov = f[o, v] + np.einsum("mnef,fn->me", g[o, o, v, v], t1, optimize=True)

    h_oo = chi_oo + np.einsum("me,ei->mi", h_ov, t1, optimize=True)

    h_ooov = g[o, o, o, v] + np.einsum("mnfe,fi->mnie", g[o, o, v, v], t1, optimize=True)

    h_vovv = g[v, o, v, v] - np.einsum("mnfe,an->amef", g[o, o, v, v], t1, optimize=True)

    singles_res = -np.einsum("mi,am->ai", h_oo, t1, optimize=True)
    singles_res += np.einsum("ae,ei->ai", chi_vv, t1, optimize=True)
    singles_res += np.einsum("anif,fn->ai", g[v, o, o, v], t1, optimize=True)
    singles_res += np.einsum("me,aeim->ai", h_ov, t2, optimize=True)
    singles_res -= 0.5 * np.einsum("mnif,afmn->ai", h_ooov, t2, optimize=True)
    singles_res += 0.5 * np.einsum("anef,efin->ai", h_vovv, t2, optimize=True)
    singles_res += 0.25 * np.einsum("mnef,aefimn->ai", g[o, o, v, v], t3, optimize=True)

    singles_res += f[v, o]

    return singles_res


def doubles_residual(t1, t2, t3, f, g, o, v, corr_slice, ints_slice):
    """Compute the projection of the CCSDT Hamiltonian on doubles
        X[a, b, i, j] = < ijab | (H_N exp(T1+T2+T3))_C | 0 >
    """

    H1, H2 = get_ccs_intermediates(t1, f, g, o, v)

    # intermediates
    I_oo = H1[o, o] + 0.5 * np.einsum("mnef,efin->mi", g[o, o, v, v], t2, optimize=True)

    I_vv = H1[v, v] - 0.5 * np.einsum("mnef,afmn->ae", g[o, o, v, v], t2, optimize=True)

    I_voov = H2[v, o, o, v] + 0.5 * np.einsum("mnef,afin->amie", g[o, o, v, v], t2, optimize=True)

    I_oooo = H2[o, o, o, o] + 0.5 * np.einsum("mnef,efij->mnij", g[o, o, v, v], t2, optimize=True)

    I_vooo = H2[v, o, o, o] + 0.5 * np.einsum('anef,efij->anij', g[v, o, v, v] + 0.5 * H2[v, o, v, v], t2, optimize=True)

    tau = 0.5 * t2 + np.einsum('ai,bj->abij', t1, t1, optimize=True)

    doubles_res = -0.5 * np.einsum("amij,bm->abij", I_vooo, t1, optimize=True)
    doubles_res += 0.5 * np.einsum("abie,ej->abij", H2[v, v, o, v], t1, optimize=True)
    doubles_res += 0.5 * np.einsum("ae,ebij->abij", I_vv, t2, optimize=True)
    doubles_res -= 0.5 * np.einsum("mi,abmj->abij", I_oo, t2, optimize=True)
    doubles_res += np.einsum("amie,ebmj->abij", I_voov, t2, optimize=True)
    doubles_res += 0.25 * np.einsum("abef,efij->abij", g[v, v, v, v], tau, optimize=True)
    doubles_res += 0.125 * np.einsum("mnij,abmn->abij", I_oooo, t2, optimize=True)
    doubles_res += 0.25 * np.einsum("me,abeijm->abij", H1[o, v], t3, optimize=True)
    doubles_res -= 0.25 * np.einsum("mnif,abfmjn->abij", g[o, o, o, v] + H2[o, o, o, v], t3, optimize=True)
    doubles_res += 0.25 * np.einsum("anef,ebfijn->abij", g[v, o, v, v] + H2[v, o, v, v], t3, optimize=True)

    doubles_res -= np.transpose(doubles_res, (1, 0, 2, 3))
    doubles_res -= np.transpose(doubles_res, (0, 1, 3, 2))

    doubles_res += g[v, v, o, o]

    return doubles_res

def triples_residual(t1, t2, t3, f, g, o, v, corr_slice, ints_slice):
    """Compute the projection of the CCSDT Hamiltonian on triples
        X[a, b, c, i, j, k] = < ijkabc | (H_N exp(T1+T2+T3))_C | 0 >
    """

    H1, H2 = get_ccsd_intermediates(t1, t2, f, g, o, v)

    I_vvov = H2[v, v, o, v] + (
              -0.5 * np.einsum("mnef,abfimn->abie", g[o, o, v, v], t3, optimize=True)
              +np.einsum("me,abim->abie", H1[o, v], t2, optimize=True)
    )
    I_vooo = H2[v, o, o, o] + 0.5 * np.einsum("mnef,aefijn->amij", g[o, o, v, v], t3, optimize=True)

    triples_res = -0.25 * np.einsum("amij,bcmk->abcijk", I_vooo, t2, optimize=True)
    triples_res += 0.25 * np.einsum("abie,ecjk->abcijk", I_vvov, t2, optimize=True)

    # Diagram 1: -A(ij) h(mi) t(AbcmjK)
    #            -h(MK) t(AbcijM)


    triples_res -= (1.0 / 12.0) * np.einsum("mK,Abcijm->AbcijK", H1[ints_slice["h"], ints_slice["H"]], t3[corr_slice["P"], :, :, :, :, corr_slice["h"]], optimize=True)
    triples_res -= (1.0 / 12.0) * np.einsum("mk,AbcjmI->AbcjkI", H1[o, o], t3[corr_slice["P"], :, :, :, :, corr_slice["h"]], optimize=True)



    triples_res += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H1[v, v], t3, optimize=True)
    triples_res += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H2[o, o, o, o], t3, optimize=True)
    triples_res += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H2[v, v, v, v], t3, optimize=True)
    triples_res += 0.25 * np.einsum("cmke,abeijm->abcijk", H2[v, o, o, v], t3, optimize=True)

    triples_res -= np.transpose(triples_res, (0, 1, 2, 3, 5, 4)) # (jk)
    triples_res -= np.transpose(triples_res, (0, 1, 2, 4, 3, 5)) + np.transpose(triples_res, (0, 1, 2, 5, 4, 3)) # (i/jk)
    triples_res -= np.transpose(triples_res, (0, 2, 1, 3, 4, 5)) # (bc)
    triples_res -= np.transpose(triples_res, (2, 1, 0, 3, 4, 5)) + np.transpose(triples_res, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return triples_res


def kernel(fock, g, o, v, maxit, convergence, diis_size, n_start_diis, out_of_core, energy_shift, nacto, nactu):
    """Solve the CCSDt system of nonlinear equations using Jacobi iterations
    with DIIS acceleration. The initial values of the T amplitudes are taken to be 0."""

    corr_slice, ints_slice = active_slice(o, v, nacto, nactu)

    eps = np.kron(np.diagonal(fock)[::2], np.ones(2))
    n = np.newaxis
    e_AbcijK = 1.0 / (- eps[ints_slice["P"], n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                      + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, ints_slice["H"]] + energy_shift )
    e_abij = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o] + energy_shift )
    e_ai = 1.0 / (-eps[v, n] + eps[n, o] + energy_shift )

    nunocc, nocc = e_ai.shape
    n1 = nocc * nunocc
    n2 = nocc**2 * nunocc**2
    n3 = nacto * nocc*2 * nactu * nunocc**2
    ndim = n1 + n2 + n3

    diis_engine = DIIS(ndim, diis_size, out_of_core)

    t1 = np.zeros((nunocc, nocc))
    t2 = np.zeros((nunocc, nunocc, nocc, nocc))
    t3 = np.zeros((nactu, nunocc, nunocc, nocc, nocc, nacto))

    old_energy = cc_energy(t1, t2, fock, g, o, v)

    print("    ==> CCSDt amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(maxit):

        tic = time.time()

        residual_singles = singles_residual(t1, t2, t3, fock, g, o, v, corr_slice, ints_slice)
        residual_doubles = doubles_residual(t1, t2, t3, fock, g, o, v, corr_slice, ints_slice)
        residual_triples = triples_residual(t1, t2, t3, fock, g, o, v, corr_slice, ints_slice)

        res_norm = np.linalg.norm(residual_singles) + np.linalg.norm(residual_doubles) + np.linalg.norm(residual_triples)

        t1 += residual_singles * e_ai
        t2 += residual_doubles * e_abij
        t3 += residual_triples * e_AbcijK

        current_energy = cc_energy(t1, t2, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < convergence and res_norm < convergence:
            break
 
        if idx >= n_start_diis:
            diis_engine.push( (t1, t2, t3), (residual_singles, residual_doubles, residual_triples), idx) 

        if idx >= diis_size + n_start_diis:
            T_extrap = diis_engine.extrapolate()
            t1 = T_extrap[:n1].reshape((nunocc, nocc))
            t2 = T_extrap[n1:n1+n2].reshape((nunocc, nunocc, nocc, nocc))
            t3 = T_extrap[n1+n2:].reshape((nactu, nunocc, nunocc, nocc, nocc, nacto))

        old_energy = current_energy

        toc = time.time()
        minutes, seconds = divmod(toc - tic, 60)
        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}    {:.2f}m {:.2f}s".format(idx, current_energy, delta_e, res_norm, minutes, seconds))
    else:
        raise ValueError("CCSDt iterations did not converge")

    diis_engine.cleanup()
    e_corr = cc_energy(t1, t2, fock, g, o, v)

    return (t1, t2, t3), e_corr


def build_100001(dT, t2, t3, H1, H2, o, v, corr_slice, ints_slice):

    H, h, P, p = corr_slices.values()
    Hi, hi, Pi, pi = ints_slice.values()


    # MM(2,3)
    dT[P, p, p, h, h, H] = (1.0 / 4.0) * (
            -1.0 * np.einsum('Amij,bcmK->AbcijK', H2.vooo[Va, :, oa, oa], t2[va, va, :, Oa], optimize=True)
    )
    dT[P, p, p, h, h, H] += (2.0 / 4.0) * (
            +1.0 * np.einsum('bmij,AcmK->AbcijK', H2.vooo[va, :, oa, oa], t2[Va, va, :, Oa], optimize=True)
    )
    dT[P, p, p, h, h, H] += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmKj,bcmi->AbcijK', H2.vooo[Va, :, Oa, oa], t2[va, va, :, oa], optimize=True)
    )
    dT[P, p, p, h, h, H] += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmKj,Acmi->AbcijK', H2.vooo[va, :, Oa, oa], t2[Va, va, :, oa], optimize=True)
    )
    dT[P, p, p, h, h, H] += (4.0 / 4.0) * (
            +1.0 * np.einsum('Abie,ecjK->AbcijK', H2.vvov[Va, va, oa, :], t2[:, va, oa, Oa], optimize=True)
    )
    dT[P, p, p, h, h, H] += (2.0 / 4.0) * (
            -1.0 * np.einsum('cbie,eAjK->AbcijK', H2.vvov[va, va, oa, :], t2[:, Va, oa, Oa], optimize=True)
    )
    dT[P, p, p, h, h, H] += (2.0 / 4.0) * (
            -1.0 * np.einsum('AbKe,ecji->AbcijK', H2.vvov[Va, va, Oa, :], t2[:, va, oa, oa], optimize=True)
    )
    dT[P, p, p, h, h, H] += (1.0 / 4.0) * (
            +1.0 * np.einsum('cbKe,eAji->AbcijK', H2.vvov[va, va, Oa, :], t2[:, Va, oa, oa], optimize=True)
    )
    # (H(2) * T3)_C
    dT[P, p, p, h, h, H] += (2.0 / 4.0) * (
            +1.0 * np.einsum('mi,AcbmjK->AbcijK', H1.oo[oa, oa], t2a.VvvooO, optimize=True)
            - 1.0 * np.einsum('Mi,AcbjMK->AbcijK', H1.oo[Oa, oa], t2a.VvvoOO, optimize=True)
    )
    dT[P, p, p, h, h, H] += (1.0 / 4.0) * (
            -1.0 * np.einsum('MK,AcbjiM->AbcijK', H1.oo[Oa, Oa], t2a.VvvooO, optimize=True)
    )
    dT[P, p, p, h, h, H] += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbijK->AbcijK', H1.vv[Va, Va], t2a.VvvooO, optimize=True)
    )
    dT[P, p, p, h, h, H] += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceijK->AbcijK', H1.vv[va, va], t2a.VvvooO, optimize=True)
            + 1.0 * np.einsum('bE,AEcijK->AbcijK', H1.vv[va, Va], t2a.VVvooO, optimize=True)
    )
    dT[P, p, p, h, h, H] += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,AcbmnK->AbcijK', H2.oooo[oa, oa, oa, oa], t2a.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mnij,AcbnMK->AbcijK', H2.oooo[Oa, oa, oa, oa], t2a.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNij,AcbMNK->AbcijK', H2.oooo[Oa, Oa, oa, oa], t2a.VvvOOO, optimize=True)
    )
    dT[P, p, p, h, h, H] += (2.0 / 4.0) * (
            +1.0 * np.einsum('MnKj,AcbniM->AbcijK', H2.oooo[Oa, oa, Oa, oa], t2a.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNKj,AcbiMN->AbcijK', H2.oooo[Oa, Oa, Oa, oa], t2a.VvvoOO, optimize=True)
    )
    #t1 = time.time()
    dT[P, p, p, h, h, H] += (2.0 / 4.0) * (
            +1.0 * np.einsum('AbeF,FceijK->AbcijK', H2.vvvv[Va, va, va, Va], t2a.VvvooO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcijK->AbcijK', H2.vvvv[Va, va, Va, Va], t2a.VVvooO, optimize=True)
    )
    dT[P, p, p, h, h, H] += (1.0 / 4.0) * (
            +0.5 * np.einsum('cbef,AfeijK->AbcijK', H2.vvvv[va, va, va, va], t2a.VvvooO, optimize=True)
            + 1.0 * np.einsum('cbeF,AFeijK->AbcijK', H2.vvvv[va, va, va, Va], t2a.VVvooO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEijK->AbcijK', H2.vvvv[va, va, Va, Va], t2a.VVVooO, optimize=True)
    )
    #print('Time for t3a vvvv =', time.time() - t1)
    dT[P, p, p, h, h, H] += (2.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,EcbmjK->AbcijK', H2.voov[Va, oa, oa, Va], t2a.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMiE,EcbjMK->AbcijK', H2.voov[Va, Oa, oa, Va], t2a.VvvoOO, optimize=True)
    )
    dT[P, p, p, h, h, H] += (4.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AcemjK->AbcijK', H2.voov[va, oa, oa, va], t2a.VvvooO, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmjK->AbcijK', H2.voov[va, oa, oa, Va], t2a.VVvooO, optimize=True)
            + 1.0 * np.einsum('bMie,AcejMK->AbcijK', H2.voov[va, Oa, oa, va], t2a.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bMiE,AEcjMK->AbcijK', H2.voov[va, Oa, oa, Va], t2a.VVvoOO, optimize=True)
    )
    dT[P, p, p, h, h, H] += (1.0 / 4.0) * (
            +1.0 * np.einsum('AMKE,EcbjiM->AbcijK', H2.voov[Va, Oa, Oa, Va], t2a.VvvooO, optimize=True)
    )
    dT[P, p, p, h, h, H] += (2.0 / 4.0) * (
            +1.0 * np.einsum('bMKe,AcejiM->AbcijK', H2.voov[va, Oa, Oa, va], t2a.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMKE,AEcjiM->AbcijK', H2.voov[va, Oa, Oa, Va], t2a.VVvooO, optimize=True)
    )

    dT[P, p, p, h, h, H] -= np.transpose(dT[P, p, p, h, h, H], (0, 2, 1, 3, 4, 5))
    dT[P, p, p, h, h, H] -= np.transpose(dT[P, p, p, h, h, H], (0, 1, 2, 4, 3, 5))

    return dT

def build_100011():
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dt2a.VvvoOO = (2.0 / 4.0) * (
            -1.0 * np.einsum('AmiJ,bcmK->AbciJK', H2.vooo[Va, :, oa, Oa], t2[va, va, :, Oa], optimize=True)
    )
    dt2a.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bmiJ,AcmK->AbciJK', H2.vooo[va, :, oa, Oa], t2[Va, va, :, Oa], optimize=True)
    )
    dt2a.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('AmKJ,bcmi->AbciJK', H2.vooo[Va, :, Oa, Oa], t2[va, va, :, oa], optimize=True)
    )
    dt2a.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bmKJ,Acmi->AbciJK', H2.vooo[va, :, Oa, Oa], t2[Va, va, :, oa], optimize=True)
    )
    dt2a.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Abie,ecJK->AbciJK', H2.vvov[Va, va, oa, :], t2[:, va, Oa, Oa], optimize=True)
    )
    dt2a.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cbie,eAJK->AbciJK', H2.vvov[va, va, oa, :], t2[:, Va, Oa, Oa], optimize=True)
    )
    dt2a.VvvoOO += (4.0 / 4.0) * (
            -1.0 * np.einsum('AbJe,eciK->AbciJK', H2.vvov[Va, va, Oa, :], t2[:, va, oa, Oa], optimize=True)
    )
    dt2a.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cbJe,eAiK->AbciJK', H2.vvov[va, va, Oa, :], t2[:, Va, oa, Oa], optimize=True)
    )
    # (H(2) * T3)_C
    dt2a.VvvoOO += (1.0 / 4.0) * (
            +1.0 * np.einsum('mi,AcbmJK->AbciJK', H1.oo[oa, oa], t2a.VvvoOO, optimize=True)
            + 1.0 * np.einsum('Mi,AcbMJK->AbciJK', H1.oo[Oa, oa], t2a.VvvOOO, optimize=True)
    )
    dt2a.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mJ,AcbmiK->AbciJK', H1.oo[oa, Oa], t2a.VvvooO, optimize=True)
            + 1.0 * np.einsum('MJ,AcbiMK->AbciJK', H1.oo[Oa, Oa], t2a.VvvoOO, optimize=True)
    )
    dt2a.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AE,EcbiJK->AbciJK', H1.vv[Va, Va], t2a.VvvoOO, optimize=True)
    )
    dt2a.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('be,AceiJK->AbciJK', H1.vv[va, va], t2a.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bE,AEciJK->AbciJK', H1.vv[va, Va], t2a.VVvoOO, optimize=True)
    )
    dt2a.VvvoOO += (2.0 / 4.0) * (
            -0.5 * np.einsum('mniJ,AcbmnK->AbciJK', H2.oooo[oa, oa, oa, Oa], t2a.VvvooO, optimize=True)
            - 1.0 * np.einsum('mNiJ,AcbmNK->AbciJK', H2.oooo[oa, Oa, oa, Oa], t2a.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,AcbMNK->AbciJK', H2.oooo[Oa, Oa, oa, Oa], t2a.VvvOOO, optimize=True)
    )
    dt2a.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('mNKJ,AcbmiN->AbciJK', H2.oooo[oa, Oa, Oa, Oa], t2a.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,AcbiMN->AbciJK', H2.oooo[Oa, Oa, Oa, Oa], t2a.VvvoOO, optimize=True)
    )
    dt2a.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AbEf,EcfiJK->AbciJK', H2.vvvv[Va, va, Va, va], t2a.VvvoOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEciJK->AbciJK', H2.vvvv[Va, va, Va, Va], t2a.VVvoOO, optimize=True)
    )
    dt2a.VvvoOO += (1.0 / 4.0) * (
            +0.5 * np.einsum('cbef,AfeiJK->AbciJK', H2.vvvv[va, va, va, va], t2a.VvvoOO, optimize=True)
            - 1.0 * np.einsum('cbEf,AEfiJK->AbciJK', H2.vvvv[va, va, Va, va], t2a.VVvoOO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEiJK->AbciJK', H2.vvvv[va, va, Va, Va], t2a.VVVoOO, optimize=True)
    )
    dt2a.VvvoOO += (1.0 / 4.0) * (
            -1.0 * np.einsum('AmiE,EcbmJK->AbciJK', H2.voov[Va, oa, oa, Va], t2a.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,EcbMJK->AbciJK', H2.voov[Va, Oa, oa, Va], t2a.VvvOOO, optimize=True)
    )
    dt2a.VvvoOO += (2.0 / 4.0) * (
            -1.0 * np.einsum('bmie,AcemJK->AbciJK', H2.voov[va, oa, oa, va], t2a.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bMie,AceMJK->AbciJK', H2.voov[va, Oa, oa, va], t2a.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bmiE,AEcmJK->AbciJK', H2.voov[va, oa, oa, Va], t2a.VVvoOO, optimize=True)
            + 1.0 * np.einsum('bMiE,AEcMJK->AbciJK', H2.voov[va, Oa, oa, Va], t2a.VVvOOO, optimize=True)
    )
    dt2a.VvvoOO += (2.0 / 4.0) * (
            +1.0 * np.einsum('AmJE,EcbmiK->AbciJK', H2.voov[Va, oa, Oa, Va], t2a.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,EcbiMK->AbciJK', H2.voov[Va, Oa, Oa, Va], t2a.VvvoOO, optimize=True)
    )
    dt2a.VvvoOO += (4.0 / 4.0) * (
            +1.0 * np.einsum('bmJe,AcemiK->AbciJK', H2.voov[va, oa, Oa, va], t2a.VvvooO, optimize=True)
            - 1.0 * np.einsum('bMJe,AceiMK->AbciJK', H2.voov[va, Oa, Oa, va], t2a.VvvoOO, optimize=True)
            - 1.0 * np.einsum('bmJE,AEcmiK->AbciJK', H2.voov[va, oa, Oa, Va], t2a.VVvooO, optimize=True)
            + 1.0 * np.einsum('bMJE,AEciMK->AbciJK', H2.voov[va, Oa, Oa, Va], t2a.VVvoOO, optimize=True)
    )

    dt2a.VvvoOO -= np.transpose(dt2a.VvvoOO, (0, 2, 1, 3, 4, 5))
    dt2a.VvvoOO -= np.transpose(dt2a.VvvoOO, (0, 1, 2, 3, 5, 4))

    return dT

def build_100111(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)
    # MM(2,3)
    dt2a.VvvOOO = (3.0 / 12.0) * (
            -1.0 * np.einsum('AmIJ,bcmK->AbcIJK', H2.vooo[Va, :, Oa, Oa], t2[va, va, :, Oa], optimize=True)
    )
    dt2a.VvvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('bmIJ,AcmK->AbcIJK', H2.vooo[va, :, Oa, Oa], t2[Va, va, :, Oa], optimize=True)
    )
    dt2a.VvvOOO += (6.0 / 12.0) * (
            +1.0 * np.einsum('AbIe,ecJK->AbcIJK', H2.vvov[Va, va, Oa, :], t2[:, va, Oa, Oa], optimize=True)
    )
    dt2a.VvvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('cbIe,eAJK->AbcIJK', H2.vvov[va, va, Oa, :], t2[:, Va, Oa, Oa], optimize=True)
    )
    # (H(2) * T3)_C
    dt2a.VvvOOO += (3.0 / 12.0) * (
            +1.0 * np.einsum('mI,AcbmJK->AbcIJK', H1.oo[oa, Oa], t2a.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MI,AcbMJK->AbcIJK', H1.oo[Oa, Oa], t2a.VvvOOO, optimize=True)
    )
    dt2a.VvvOOO += (1.0 / 12.0) * (
            -1.0 * np.einsum('AE,EcbIJK->AbcIJK', H1.vv[Va, Va], t2a.VvvOOO, optimize=True)
    )
    dt2a.VvvOOO += (2.0 / 12.0) * (
            -1.0 * np.einsum('be,AceIJK->AbcIJK', H1.vv[va, va], t2a.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bE,AEcIJK->AbcIJK', H1.vv[va, Va], t2a.VVvOOO, optimize=True)
    )
    dt2a.VvvOOO += (3.0 / 12.0) * (
            -0.5 * np.einsum('mnIJ,AcbmnK->AbcIJK', H2.oooo[oa, oa, Oa, Oa], t2a.VvvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,AcbnMK->AbcIJK', H2.oooo[Oa, oa, Oa, Oa], t2a.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,AcbMNK->AbcIJK', H2.oooo[Oa, Oa, Oa, Oa], t2a.VvvOOO, optimize=True)
    )
    dt2a.VvvOOO += (2.0 / 12.0) * (
            -1.0 * np.einsum('AbEf,EcfIJK->AbcIJK', H2.vvvv[Va, va, Va, va], t2a.VvvOOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcIJK->AbcIJK', H2.vvvv[Va, va, Va, Va], t2a.VVvOOO, optimize=True)
    )
    dt2a.VvvOOO += (1.0 / 12.0) * (
            +0.5 * np.einsum('cbef,AfeIJK->AbcIJK', H2.vvvv[va, va, va, va], t2a.VvvOOO, optimize=True)
            - 1.0 * np.einsum('cbEf,AEfIJK->AbcIJK', H2.vvvv[va, va, Va, va], t2a.VVvOOO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEIJK->AbcIJK', H2.vvvv[va, va, Va, Va], t2a.VVVOOO, optimize=True)
    )
    dt2a.VvvOOO += (3.0 / 12.0) * (
            -1.0 * np.einsum('AmIE,EcbmJK->AbcIJK', H2.voov[Va, oa, Oa, Va], t2a.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,EcbMJK->AbcIJK', H2.voov[Va, Oa, Oa, Va], t2a.VvvOOO, optimize=True)
    )
    dt2a.VvvOOO += (6.0 / 12.0) * (
            -1.0 * np.einsum('bmIe,AcemJK->AbcIJK', H2.voov[va, oa, Oa, va], t2a.VvvoOO, optimize=True)
            + 1.0 * np.einsum('bmIE,AEcmJK->AbcIJK', H2.voov[va, oa, Oa, Va], t2a.VVvoOO, optimize=True)
            - 1.0 * np.einsum('bMIe,AceMJK->AbcIJK', H2.voov[va, Oa, Oa, va], t2a.VvvOOO, optimize=True)
            + 1.0 * np.einsum('bMIE,AEcMJK->AbcIJK', H2.voov[va, Oa, Oa, Va], t2a.VVvOOO, optimize=True)
    )

    dt2a.VvvOOO -= np.transpose(dt2a.VvvOOO, (0, 1, 2, 4, 3, 5)) + np.transpose(dt2a.VvvOOO, (0, 1, 2, 3, 5, 4)) \
           + np.transpose(dt2a.VvvOOO, (0, 1, 2, 5, 4, 3)) - np.transpose(dt2a.VvvOOO, (0, 1, 2, 4, 5, 3)) \
           - np.transpose(dt2a.VvvOOO, (0, 1, 2, 5, 3, 4))

    dt2a.VvvOOO -= np.transpose(dt2a.VvvOOO, (0, 2, 1, 3, 4, 5))

    return dT

def build_110001(T, dT, H, system):
    oa, Oa, va, Va, ob, Ob, vb, Vb = get_active_slices(system)

    # MM(2,3)
    dt2a.VVvooO = (2.0 / 4.0) * (
            -1.0 * np.einsum('Amij,BcmK->ABcijK', H2.vooo[Va, :, oa, oa], t2[P, p, :, H], optimize=True)
    )
    dt2a.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('cmij,BAmK->ABcijK', H2.vooo[va, :, oa, oa], t2[P, P, :, H], optimize=True)
    )
    dt2a.VVvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('AmKj,Bcmi->ABcijK', H2.vooo[Va, :, Oa, oa], t2[P, p, :, h], optimize=True)
    )
    dt2a.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('cmKj,BAmi->ABcijK', H2.vooo[va, :, Oa, oa], t2[P, P, :, h], optimize=True)
    )
    dt2a.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('ABie,ecjK->ABcijK', H2.vvov[Va, Va, oa, :], t2[:, p, h, H], optimize=True)
    )
    dt2a.VVvooO += (4.0 / 4.0) * (
            -1.0 * np.einsum('cBie,eAjK->ABcijK', H2.vvov[va, Va, oa, :], t2[:, P, h, H], optimize=True)
    )
    dt2a.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ABKe,ecji->ABcijK', H2.vvov[Va, Va, Oa, :], t2[:, p, h, h], optimize=True)
    )
    dt2a.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cBKe,eAji->ABcijK', H2.vvov[va, Va, Oa, :], t2[:, P, h, h], optimize=True)
    )
    # (H(2) * T3)_C
    dt2a.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('mi,BAcmjK->ABcijK', H1.oo[oa, oa], t2a.VVvooO, optimize=True)
            - 1.0 * np.einsum('Mi,BAcjMK->ABcijK', H1.oo[Oa, oa], t2a.VVvoOO, optimize=True)
    )
    dt2a.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('MK,BAcjiM->ABcijK', H1.oo[Oa, Oa], t2a.VVvooO, optimize=True)
    )
    dt2a.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('Ae,BceijK->ABcijK', H1.vv[Va, va], t2a.VvvooO, optimize=True)
            - 1.0 * np.einsum('AE,BEcijK->ABcijK', H1.vv[Va, Va], t2a.VVvooO, optimize=True)
    )
    dt2a.VVvooO += (1.0 / 4.0) * (
            +1.0 * np.einsum('ce,ABeijK->ABcijK', H1.vv[va, va], t2a.VVvooO, optimize=True)
            + 1.0 * np.einsum('cE,ABEijK->ABcijK', H1.vv[va, Va], t2a.VVVooO, optimize=True)
    )
    dt2a.VVvooO += (1.0 / 4.0) * (
            -0.5 * np.einsum('mnij,BAcmnK->ABcijK', H2.oooo[oa, oa, oa, oa], t2a.VVvooO, optimize=True)
            - 1.0 * np.einsum('mNij,BAcmNK->ABcijK', H2.oooo[oa, Oa, oa, oa], t2a.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNij,BAcMNK->ABcijK', H2.oooo[Oa, Oa, oa, oa], t2a.VVvOOO, optimize=True)
    )
    dt2a.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('mNKj,BAcmiN->ABcijK', H2.oooo[oa, Oa, Oa, oa], t2a.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNKj,BAciMN->ABcijK', H2.oooo[Oa, Oa, Oa, oa], t2a.VVvoOO, optimize=True)
    )
    dt2a.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('ABEf,EcfijK->ABcijK', H2.vvvv[Va, Va, Va, va], t2a.VvvooO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcijK->ABcijK', H2.vvvv[Va, Va, Va, Va], t2a.VVvooO, optimize=True)
    )
    dt2a.VVvooO += (2.0 / 4.0) * (
            +0.5 * np.einsum('cBef,AfeijK->ABcijK', H2.vvvv[va, Va, va, va], t2a.VvvooO, optimize=True)
            - 1.0 * np.einsum('cBEf,AEfijK->ABcijK', H2.vvvv[va, Va, Va, va], t2a.VVvooO, optimize=True)
            + 0.5 * np.einsum('cBEF,AFEijK->ABcijK', H2.vvvv[va, Va, Va, Va], t2a.VVVooO, optimize=True)
    )
    dt2a.VVvooO += (4.0 / 4.0) * (
            +1.0 * np.einsum('Amie,BcemjK->ABcijK', H2.voov[Va, oa, oa, va], t2a.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMie,BcejMK->ABcijK', H2.voov[Va, Oa, oa, va], t2a.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmjK->ABcijK', H2.voov[Va, oa, oa, Va], t2a.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMiE,BEcjMK->ABcijK', H2.voov[Va, Oa, oa, Va], t2a.VVvoOO, optimize=True)
    )
    dt2a.VVvooO += (2.0 / 4.0) * (
            +1.0 * np.einsum('cmie,ABemjK->ABcijK', H2.voov[va, oa, oa, va], t2a.VVvooO, optimize=True)
            - 1.0 * np.einsum('cMie,ABejMK->ABcijK', H2.voov[va, Oa, oa, va], t2a.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cmiE,ABEmjK->ABcijK', H2.voov[va, oa, oa, Va], t2a.VVVooO, optimize=True)
            - 1.0 * np.einsum('cMiE,ABEjMK->ABcijK', H2.voov[va, Oa, oa, Va], t2a.VVVoOO, optimize=True)
    )
    dt2a.VVvooO += (2.0 / 4.0) * (
            -1.0 * np.einsum('AMKe,BcejiM->ABcijK', H2.voov[Va, Oa, Oa, va], t2a.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMKE,BEcjiM->ABcijK', H2.voov[Va, Oa, Oa, Va], t2a.VVvooO, optimize=True)
    )
    dt2a.VVvooO += (1.0 / 4.0) * (
            -1.0 * np.einsum('cMKe,ABejiM->ABcijK', H2.voov[va, Oa, Oa, va], t2a.VVvooO, optimize=True)
            - 1.0 * np.einsum('cMKE,ABEjiM->ABcijK', H2.voov[va, Oa, Oa, Va], t2a.VVVooO, optimize=True)
    )

    dt2a.VVvooO -= np.transpose(dt2a.VVvooO, (1, 0, 2, 3, 4, 5))
    dt2a.VVvooO -= np.transpose(dt2a.VVvooO, (0, 1, 2, 4, 3, 5))

    return dT

