import numpy as np
from miniccpy.driver import run_scf_gamess, run_cc_calc, get_hbar, run_leftcc_calc

def build_Doo(t1, t2, l1, l2):
    Doo = -1.0 * np.einsum('ei,ej->ij', t1, l1)
    Doo -= np.einsum('efim,efjm->ij', t2, l2)
    return Doo

def build_Dvv(t1, t2, l1, l2):
    Dvv = np.einsum('bm,am->ab', t1, l1)
    Dvv += np.einsum('bemn,aemn->ab', t2, l2)
    return Dvv

def build_Dvo(l1):
    return l1.copy()

def build_Dov(t1, t2, l1, l2):
    Dov = 2.0 * t1.T.copy()
    Dov += 2.0 * np.einsum('em,aeim->ia', l1, t2)
    Dov -= np.einsum('em,aemi->ia', l1, build_tau(t1, t2))
    tmp = np.einsum('efmn,efin->mi', l2, t2)
    Dov -= np.einsum('mi,am->ia', tmp, t1)
    tmp = np.einsum('efmn,afmn->ea', l2, t2)
    Dov -= np.einsum('ea,ei->ia', tmp, t1)
    return Dov

def build_Doooo(t1, t2, l2):
    return np.einsum('efij,efkl->ijkl', build_tau(t1, t2), l2)

def build_Dvvvv(t1, t2, l2):
    return np.einsum('abmn,cdmn->abcd', build_tau(t1, t2), l2)

def build_Dooov(t1, t2, l1, l2):
    tau = build_tau(t1, t2)
    tmp = 2.0 * tau - tau.transpose(0, 1, 3, 2)
    Dooov = -1.0 * np.einsum('ek,eaij->ijka', l1, tmp)
    Dooov -= np.einsum('ei,aejk->ijka', t1, l2)

    Goo = build_Goo(t2, l2)
    Dooov -= 2.0 * np.einsum('ik,aj->ijka', Goo, t1)
    Dooov += np.einsum('jk,ai->ijka', Goo, t1)
    tmp = np.einsum('afjm,efkm->jake', t2, l2)
    Dooov -= 2.0 * np.einsum('jake,ei->ijka', tmp, t1)
    Dooov += np.einsum('iake,ej->ijka', tmp, t1)

    tmp = np.einsum('efij,efkm->ijkm', t2, l2)
    Dooov += np.einsum('ijkm,am->ijka', tmp, t1)
    tmp = np.einsum('afmj,efkm->jake', t2, l2)
    Dooov += np.einsum('jake,ei->ijka', tmp, t1)
    tmp = np.einsum('eaim,efkm->iakf', t2, l2)
    Dooov += np.einsum('iakf,fj->ijka', tmp, t1)

    tmp = np.einsum('efkm,fj->kmej', l2, t1)
    tmp = np.einsum('kmej,ei->kmij', tmp, t1)
    Dooov += np.einsum('kmij,am->ijka', tmp, t1)
    return Dooov

def build_Dvvvo(t1, t2, l1, l2):
    tau = build_tau(t1, t2)
    tmp = 2.0 * tau - tau.transpose(0, 1, 3, 2)
    Dvvvo = np.einsum('cm,abmi->abci', l1, tmp)
    Dvvvo += np.einsum('am,bcim->abci', t1, l2)
        
    Gvv = build_Gvv(t2, l2)
    Dvvvo -= 2.0 * np.einsum('ca,bi->abci', Gvv, t1)
    Dvvvo += np.einsum('cb,ai->abci', Gvv, t1)
    tmp = np.einsum('beim,cenm->ibnc', t2, l2)
    Dvvvo += 2.0 * np.einsum('ibnc,an->abci', tmp, t1)
    Dvvvo -= np.einsum('ianc,bn->abci', tmp, t1)

    tmp = np.einsum('abnm,cenm->abce', t2, l2)
    Dvvvo -= np.einsum('abce,ei->abci', tmp, t1)
    tmp = np.einsum('aeni,cenm->iamc', t2, l2)
    Dvvvo -= np.einsum('iamc,bm->abci', tmp, t1)
    tmp = np.einsum('bemi,cenm->ibnc', t2, l2)
    Dvvvo -= np.einsum('ibnc,an->abci', tmp, t1)

    tmp = np.einsum('cenm,ei->nmci', l2, t1)
    tmp = np.einsum('nmci,an->amci', tmp, t1)
    Dvvvo -= np.einsum('amci,bm->abci', tmp, t1)
    return Dvvvo

def build_Dovov(t1, t2, l1, l2):
    Dovov = -1.0 * np.einsum('ai,bj->iajb', t1, l1)
    Dovov -= np.einsum('bemi,eajm->iajb', build_tau(t1, t2), l2)
    Dovov -= np.einsum('beim,eamj->iajb', t2, l2)
    return Dovov

def build_Doovv(t1, t2, l1, l2):

    tau = build_tau(t1, t2)
    # spin-summed tau
    tau_ss = 2.0 * tau - tau.transpose(0, 1, 3, 2)

    Doovv = 4.0 * np.einsum('ai,bj->ijab', t1, l1)
    Doovv += 2.0 * tau_ss.transpose(2, 3, 0, 1)
    Doovv += l2.transpose(2, 3, 0, 1)

    tmp1 = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    tmp2 = 2.0 * np.einsum('em,bejm->jb', l1, tmp1)
    Doovv += 2.0 * np.einsum('jb,ai->ijab', tmp2, t1)
    Doovv -= np.einsum('ja,bi->ijab', tmp2, t1)
    tmp2 = 2.0 * np.einsum('ebij,em->ijmb', tmp1, l1)
    Doovv -= np.einsum('ijmb,am->ijab', tmp2, t1)
    tmp2 = 2.0 * np.einsum('bajm,em->jeba', tau_ss, l1)
    Doovv -= np.einsum('jeba,ei->ijab', tmp2, t1)

    Doovv += 4.0 * np.einsum('aeim,ebmj->ijab', t2, l2)
    Doovv -= 2.0 * np.einsum('bemj,aeim->ijab', tau, l2)

    tmp_oooo = np.einsum('efij,efmn->ijmn', t2, l2)
    Doovv += np.einsum('ijmn,abmn->ijab', tmp_oooo, t2)
    tmp1 = np.einsum('bfnj,efmn->jbme', t2, l2)
    Doovv += np.einsum('jbme,aemi->ijab', tmp1, t2)
    tmp1 = np.einsum('fbim,efmn->ibne', t2, l2)
    Doovv += np.einsum('ibne,aenj->ijab', tmp1, t2)
    Gvv = build_Gvv(t2, l2)
    Doovv += 4.0 * np.einsum('eb,aeij->ijab', Gvv, tau)
    Doovv -= 2.0 * np.einsum('ea,beij->ijab', Gvv, tau)
    Goo = build_Goo(t2, l2)
    Doovv -= 4.0 * np.einsum('jm,abim->ijab', Goo, tau) 
    Doovv += 2.0 * np.einsum('jm,baim->ijab', Goo, tau)
    tmp1 = np.einsum('afin,efmn->iame', t2, l2)
    Doovv -= 4.0 * np.einsum('iame,bemj->ijab', tmp1, tau)
    Doovv += 2.0 * np.einsum('ibme,aemj->ijab', tmp1, tau)
    Doovv += 4.0 * np.einsum('jbme,aeim->ijab', tmp1, t2)
    Doovv -= 2.0 * np.einsum('jame,beim->ijab', tmp1, t2)

    # this can definitely be optimized better
    tmp = np.einsum('bn,ijmn->ijmb', t1, tmp_oooo)
    Doovv += np.einsum('am,ijmb->ijab', t1, tmp)
    tmp = np.einsum('ei,efmn->mnif', t1, l2)
    tmp = np.einsum('fj,mnif->mnij', t1, tmp)
    Doovv += np.einsum('mnij,abmn->ijab', tmp, t2)
    tmp = np.einsum('ei,efmn->mnif', t1, l2)
    tmp = np.einsum('mnif,bfnj->mijb', tmp, t2)
    Doovv += np.einsum('am,mijb->ijab', t1, tmp)
    tmp = np.einsum('fj,efmn->mnej', t1, l2)
    tmp = np.einsum('mnej,aemi->njia', tmp, t2)
    Doovv += np.einsum('bn,njia->ijab', t1, tmp)
    tmp = np.einsum('ej,efmn->mnjf', t1, l2)
    tmp = np.einsum('mnjf,fbim->njib', tmp, t2)
    Doovv += np.einsum('an,njib->ijab', t1, tmp)
    tmp = np.einsum('fi,efmn->mnei', t1, l2)
    tmp = np.einsum('mnei,aenj->mija', tmp, t2)
    Doovv += np.einsum('bm,mija->ijab', t1, tmp)

    tmp = np.einsum('fj,efmn->mnej', t1, l2)
    tmp = np.einsum('ei,mnej->mnij', t1, tmp)
    tmp = np.einsum('bn,mnij->mbij', t1, tmp)
    Doovv += np.einsum('am,mbij->ijab', t1, tmp)
    return Doovv

def build_tau(t1, t2):
    return t2 + np.einsum('ai,bj->abij', t1, t1)

def build_Gvv(t2, l2):
    return -1.0 * np.einsum('ebij,abij->ae', t2, l2)

def build_Goo(t2, l2):
    return np.einsum('abmj,abij->mi', t2, l2)

if __name__ == "__main__":

    fock, g, e_hf, o, v = run_scf_gamess("h2o.FCIDUMP", 20, 50, 0, rhf=True)

    T, E_corr = run_cc_calc(fock, g, o, v, method='rccsd', maxit=80, convergence=1.0e-010)
    H1, H2 = get_hbar(T, fock, g, o, v, method="rccsd")
    L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_rccsd", maxit=80) # set maxit=0 to return L's equal to T's (i.e., no left-CCSD iterations)

    t1, t2 = T
    l1, l2 = L

    # orbital dimensions
    nu, no = t1.shape
    # useful slicing objects
    o = slice(0, no)
    v = slice(no, no + nu)

    # 1-body RDM
    Doo = build_Doo(t1, t2, l1, l2)
    Dvv = build_Dvv(t1, t2, l1, l2)
    Dvo = build_Dvo(l1)
    Dov = build_Dov(t1, t2, l1, l2)

    # 2-body RDM
    Doooo = build_Doooo(t1, t2, l2)
    Dvvvv = build_Dvvvv(t1, t2, l2)
    Dooov = build_Dooov(t1, t2, l1, l2)
    Dvvvo = build_Dvvvo(t1, t2, l1, l2)
    Dovov = build_Dovov(t1, t2, l1, l2)
    Doovv = build_Doovv(t1, t2, l1, l2)

    # One-electron energy
    e_ov = np.einsum("ia,ia->", fock[o, v], Dov)
    e_vo = np.einsum("ai,ai->", fock[v, o], Dvo)
    e_oo = np.einsum('ij,ij->', fock[o, o], Doo)
    e_vv = np.einsum('ab,ab->', fock[v, v], Dvv)
    onebody = e_ov + e_vo + e_oo + e_vv
    print("")
    print("   1-body RDM")
    print("   -----------")
    print("   e_ov = ", e_ov)
    print("   e_vo = ", e_vo)
    print("   e_oo = ", e_oo)
    print("   e_vv = ", e_vv)
    print("   Total 1-electron energy = ", onebody)
    # Two-electron energy
    e_oooo = 0.5 * np.einsum('ijkl,ijkl->', g[o, o, o, o], Doooo)
    e_vvvv = 0.5 * np.einsum('abcd,abcd->', g[v, v, v, v], Dvvvv)
    e_ooov = np.einsum('ijka,ijka->', g[o, o, o, v], Dooov)
    e_vvvo = np.einsum('abci,abci->', g[v, v, v, o], Dvvvo)
    e_ovov = np.einsum('iajb,iajb->', g[o, v, o, v], Dovov)
    e_oovv = 0.5 * np.einsum('ijab,ijab->', g[o, o, v, v], Doovv)
    twobody = e_oooo + e_vvvv + e_ooov + e_vvvo + e_ovov + e_oovv
    print("")
    print("   2-body RDM")
    print("   -----------")
    print("   e_oooo = ", e_oooo)
    print("   e_vvvv = ", e_vvvv)
    print("   e_ooov = ", e_ooov)
    print("   e_vvvo = ", e_vvvo)
    print("   e_ovov = ", e_ovov)
    print("   e_oovv = ", e_oovv)
    print("   Total 2-electron energy = ", twobody)
    # Check that total CC energy from RDM matches correlation energy
    print("")
    print("   Energy Check")
    print("   ------------")
    print("   Total Correlation Energy = ", onebody + twobody)
    print("   CC Correlation Energy = ", E_corr)

    assert np.allclose(onebody + twobody, E_corr, atol=1.0e-07)

