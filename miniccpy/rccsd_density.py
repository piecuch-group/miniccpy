"""
Copyright 2021, Prof. T. Daniel Crawford, Virginia Tech
"""
import numpy as np
import time

def build_rdm1(T, L):
    t1, t2 = T
    l1, l2 = L

    nu, no = t1.shape
    o = slice(0, no)
    v = slice(no, no + nu)
    rdm1 = np.zeros((nu + no, nu + no))
    
    rdm1[o, o] = build_Doo(t1, t2, l1, l2)
    rdm1[v, v] = build_Dvv(t1, t2, l1, l2)
    rdm1[v, o] = build_Dvo(l1)
    rdm1[o, v] = build_Dov(t1, t2, l1, l2)
    return rdm1

def build_rdm2(T, L):
    t1, t2 = T
    l1, l2 = L

    nu, no = t1.shape
    o = slice(0, no)
    v = slice(no, no + nu)
    rdm2 = np.zeros((nu + no, nu + no, nu + no, nu + no))
    
    rdm2[o, o, o, o] = build_Doooo(t1, t2, l2)
    rdm2[v, v, v, v] = build_Dvvvv(t1, t2, l2)
    rdm2[o, o, o, v] = build_Dooov(t1, t2, l1, l2)
    rdm2[v, v, v, o] = build_Dvvvo(t1, t2, l1, l2)
    rdm2[o, v, o, v] = build_Dovov(t1, t2, l1, l2)
    rdm2[o, o, v, v] = build_Doovv(t1, t2, l1, l2)
    return rdm2

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
    tmp = 2.0 * build_tau(t1, t2) - build_tau(t1, t2).swapaxes(2, 3)
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
    tmp = 2.0 * build_tau(t1, t2) - build_tau(t1, t2).swapaxes(2, 3)
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
    tau_spinad = 2.0 * tau - tau.swapaxes(2,3)

    Doovv = 4.0 * np.einsum('ai,bj->ijab', t1, l1)
    Doovv += 2.0 * tau_spinad.transpose(2, 3, 0, 1)
    Doovv += l2.transpose(2, 3, 0, 1)

    tmp1 = 2.0 * t2 - t2.swapaxes(2,3)
    tmp2 = 2.0 * np.einsum('em,bejm->jb', l1, tmp1)
    Doovv += 2.0 * np.einsum('jb,ai->ijab', tmp2, t1)
    Doovv -= np.einsum('ja,bi->ijab', tmp2, t1)
    tmp2 = 2.0 * np.einsum('ebij,em->ijmb', tmp1, l1)
    Doovv -= np.einsum('ijmb,am->ijab', tmp2, t1)
    tmp2 = 2.0 * np.einsum('bajm,em->jeba', tau_spinad, l1)
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
    Doovv -= 4.0 * np.einsum('jm,abim->ijab', Goo, tau)  # use tau_spinad?
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

