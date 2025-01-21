import numpy as np

def rdm2_ccsd(L, T):
    
    t1, t2 = T
    l1, l2 = L

    #oooo Gamma_ijkl --ij--.--kl--
    rdm2_oooo = (
                +0.5*np.einsum('efij,efkl->ijkl', l2, t2, optimize=True)

                +np.einsum('efij,ek,fl->ijkl', l2, t1, t1, optimize=True)
    ) 

    #ooov Gamma_ijka --ija--.--k--
    rdm2_ooov = -np.einsum('eaij,ek->ijka', l2, t1, optimize=True)

    rdm2_oovo = -np.transpose(rdm2_ooov, (0,1,3,2))

    #oovv Gamma_ijab --ijab--.
    rdm2_oovv = np.transpose(l2, (2,3,0,1))

    #ovoo Gamma_iajk --i--.--jka--
    rdm2_ovoo = (
                    -np.einsum('ei,eajk->iajk', l1, t2, optimize=True)

                    -np.einsum('ei,ej,ak->iajk', l1, t1, t1, optimize=True)
                    +np.einsum('ei,ek,aj->iajk', l1, t1, t1, optimize=True)

                    -np.einsum('efim,ej,afkm->iajk', l2, t1, t2, optimize=True)
                    +np.einsum('efim,ek,afjm->iajk', l2, t1, t2, optimize=True)

                    -0.5*np.einsum('efmi,ak,efmj->iajk', l2, t1, t2, optimize=True) 
                    +0.5*np.einsum('efmi,aj,efmk->iajk', l2, t1, t2, optimize=True)

                    +0.5*np.einsum('efim,am,efjk->iajk', l2, t1, t2, optimize=True)

                    +np.einsum('efim,ej,fk,am->iajk', l2, t1, t1, t1, optimize=True)
    )

    #ovov Gamma_iajb --ib--.--ja--
    rdm2_ovov = (
                -np.einsum('bi,aj->iajb', l1, t1, optimize=True)

                -np.einsum('ebim,eajm->iajb', l2, t2, optimize=True)

                -np.einsum('ebim,ej,am->iajb', l2, t1, t1, optimize=True)
    )
    rdm2_ovvo = -np.transpose(rdm2_ovov, (0,1,3,2))

    #ovvv Gamma_iabc --ibc--.--a--
    rdm2_ovvv = +np.einsum('bcim,am->iabc', l2, t1, optimize=True)

    #vvoo Gamma_abij .--ijab--
    rdm2_vvoo = (
                    +np.einsum('ai,bj->abij', t1, t1, optimize=True)
                    -np.einsum('bi,aj->abij', t1, t1, optimize=True)

                    +np.einsum('em,bj,eami->abij', l1, t1, t2, optimize=True)
                    -np.einsum('em,bi,eamj->abij', l1, t1, t2, optimize=True)
                    -np.einsum('em,aj,ebmi->abij', l1, t1, t2, optimize=True)
                    +np.einsum('em,ai,ebmj->abij', l1, t1, t2, optimize=True)

                    +np.einsum('efmn,eami,bfjn->abij', l2, t2, t2, optimize=True)
                    -np.einsum('efmn,eamj,bfin->abij', l2, t2, t2, optimize=True)

                    +0.25*np.einsum('efmn,abmn,efij->abij', l2, t2, t2, optimize=True)

                    +np.einsum('em,ei,abjm->abij', l1, t1, t2, optimize=True)
                    -np.einsum('em,ej,abim->abij', l1, t1, t2, optimize=True)

                    +0.5*np.einsum('efmn,efmi,abjn->abij', l2, t2, t2, optimize=True)
                    -0.5*np.einsum('efmn,efmj,abin->abij', l2, t2, t2, optimize=True)

                    +np.einsum('em,am,beij->abij', l1, t1, t2, optimize=True)
                    -np.einsum('em,bm,aeij->abij', l1, t1, t2, optimize=True)

                    +0.5*np.einsum('efmn,eamn,bfij->abij', l2, t2, t2, optimize=True)
                    -0.5*np.einsum('efmn,ebmn,afij->abij', l2, t2, t2, optimize=True)

                    -np.einsum('em,ej,bm,ai->abij', l1, t1, t1, t1, optimize=True)
                    +np.einsum('em,ei,bm,aj->abij', l1, t1, t1, t1, optimize=True)
                    +np.einsum('em,ej,am,bi->abij', l1, t1, t1, t1, optimize=True)
                    -np.einsum('em,ei,am,bj->abij', l1, t1, t1, t1, optimize=True)

                    -np.einsum('efmn,fj,bn,eami->abij', l2, t1, t1, t2, optimize=True)
                    +np.einsum('efmn,fi,bn,eamj->abij', l2, t1, t1, t2, optimize=True)
                    +np.einsum('efmn,fj,an,ebmi->abij', l2, t1, t1, t2, optimize=True)
                    -np.einsum('efmn,fi,an,ebmj->abij', l2, t1, t1, t2, optimize=True)

                    -0.5*np.einsum('efmn,ai,ej,bfmn->abij', l2, t1, t1, t2, optimize=True)
                    +0.5*np.einsum('efmn,aj,ei,bfmn->abij', l2, t1, t1, t2, optimize=True)
                    +0.5*np.einsum('efmn,bi,ej,afmn->abij', l2, t1, t1, t2, optimize=True)
                    -0.5*np.einsum('efmn,bj,ei,afmn->abij', l2, t1, t1, t2, optimize=True)

                    -0.5*np.einsum('efmn,ai,bm,efjn->abij', l2, t1, t1, t2, optimize=True)
                    +0.5*np.einsum('efmn,aj,bm,efin->abij', l2, t1, t1, t2, optimize=True)
                    +0.5*np.einsum('efmn,bi,am,efjn->abij', l2, t1, t1, t2, optimize=True)
                    -0.5*np.einsum('efmn,bj,am,efin->abij', l2, t1, t1, t2, optimize=True)

                    +0.5*np.einsum('efmn,am,bn,efij->abij', l2, t1, t1, t2, optimize=True)

                    +0.5*np.einsum('efmn,fj,ei,abmn->abij', l2, t1, t1, t2, optimize=True)

                    +np.einsum('efmn,am,bn,ei,fj->abij', l2, t1, t1, t1, t1, optimize=True)
    )
    rdm2_vvoo += t2.copy()

    #vvov Gamma_abic --c--.--iab--
    rdm2_vvov = (
                +np.einsum('cm,abim->abic', l1, t2, optimize=True)

                +np.einsum('cm,ai,bm->abic', l1, t1, t1, optimize=True)
                -np.einsum('cm,bi,am->abic', l1, t1, t1, optimize=True)

                +np.einsum('ecmn,bn,eami->abic', l2, t1, t2, optimize=True)
                -np.einsum('ecmn,an,ebmi->abic', l2, t1, t2, optimize=True)

                +0.5*np.einsum('cemn,ai,bemn->abic', l2, t1, t2, optimize=True)
                -0.5*np.einsum('cemn,bi,aemn->abic', l2, t1, t2, optimize=True)

                -0.5*np.einsum('ecmn,ei,abmn->abic', l2, t1, t2, optimize=True)

                -np.einsum('ecmn,ei,am,bn->abic', l2, t1, t1, t1, optimize=True)
    )
    rdm2_vvvo = -np.transpose(rdm2_vvov, (0,1,3,2))

    #vvvv Gamma_abcd --cd--.--ab--
    rdm2_vvvv = (
                +0.5*np.einsum('cdmn,abmn->abcd', l2, t2, optimize=True)

                +np.einsum('cdmn,am,bn->abcd', l2, t1, t1, optimize=True)  
    ) 



    rdm2_oooa = np.concatenate((rdm2_oooo, rdm2_ooov), axis=3)
    rdm2_oova = np.concatenate((rdm2_oovo, rdm2_oovv), axis=3)
    rdm2_ooaa = np.concatenate((rdm2_oooa, rdm2_oova), axis=2)
    rdm2_ovoa = np.concatenate((rdm2_ovoo, rdm2_ovov), axis=3)
    rdm2_ovva = np.concatenate((rdm2_ovvo, rdm2_ovvv), axis=3)
    rdm2_ovaa = np.concatenate((rdm2_ovoa, rdm2_ovva), axis=2)
    rdm2_voaa = -np.transpose(rdm2_ovaa, (1,0,2,3))
    rdm2_vvoa = np.concatenate((rdm2_vvoo, rdm2_vvov), axis=3)
    rdm2_vvva = np.concatenate((rdm2_vvvo, rdm2_vvvv), axis=3)
    rdm2_vvaa = np.concatenate((rdm2_vvoa, rdm2_vvva), axis=2)
    rdm2_oaaa = np.concatenate((rdm2_ooaa, rdm2_ovaa), axis=1)
    rdm2_vaaa = np.concatenate((rdm2_voaa, rdm2_vvaa), axis=1)
    rdm2 = np.concatenate((rdm2_oaaa, rdm2_vaaa), axis=0)

    return rdm2





def rdm2_ccsd_fact(L, T):
    
    t1, t2 = T
    l1, l2 = L

    tau2_asy = ( np.einsum('ai,bj->abij', t1, t1)
                -np.einsum('bi,aj->abij', t1, t1)
    )
    tau2_asy += t2.copy()

    tau2_sy = 2*np.einsum('ai,bj->abij', t1, t1)
    tau2_sy += t2.copy()
    

    #L2
    rdm2_oovv = np.transpose(l2, (2,3,0,1))

    #tau2
    rdm2_vvoo = tau2_asy.copy()

    #L1T1
    rdm2_ovov = -np.einsum('bi,aj->iajb', l1, t1, optimize=True)

    #L1tau2
    rdm2_ovoo = -np.einsum('ei,eajk->iajk', l1, tau2_asy, optimize=True)    
    rdm2_vvov = np.einsum('cm,abim->abic', l1, tau2_asy, optimize=True)

    #L1tau2t1   
    rdm2_vvoo += -np.einsum('em,eaji,bm->abij', l1, tau2_asy, t1, optimize=True)
    rdm2_vvoo += np.einsum('em,ebji,am->abij', l1, tau2_asy, t1, optimize=True)
    rdm2_vvoo += -np.einsum('em,abim,ej->abij', l1, t2, t1, optimize=True)
    rdm2_vvoo += np.einsum('em,abjm,ei->abij', l1, t2, t1, optimize=True)
    rdm2_vvoo += np.einsum('em,bj,eami->abij', l1, t1, t2, optimize=True)
    rdm2_vvoo += -np.einsum('em,bi,eamj->abij', l1, t1, t2, optimize=True)
    rdm2_vvoo += -np.einsum('em,aj,ebmi->abij', l1, t1, t2, optimize=True)
    rdm2_vvoo += np.einsum('em,ai,ebmj->abij', l1, t1, t2, optimize=True)

    #L2T1
    rdm2_ooov = -np.einsum('eaij,ek->ijka', l2, t1, optimize=True)
    rdm2_ovvv = np.einsum('bcim,am->iabc', l2, t1, optimize=True)

    #L2tau2
    rdm2_oooo = 0.5*np.einsum('efij,efkl->ijkl', l2, tau2_sy)

    rdm2_vvvv = 0.5*np.einsum('cdmn,abmn->abcd', l2, tau2_sy, optimize=True)

    rdm2_ovov += -np.einsum('ebim,eajm->iajb', l2, t2, optimize=True)
    rdm2_ovov += -np.einsum('ebim,ej,am->iajb', l2, t1, t1, optimize=True)

    #L2tau2t1
    rdm2_ovoo += -np.einsum('efim,afkm,ej->iajk', l2, t2, t1, optimize=True)
    rdm2_ovoo += np.einsum('efim,afjm,ek->iajk', l2, t2, t1, optimize=True)
    rdm2_ovoo += 0.5*np.einsum('efim,efjk,am->iajk', l2, tau2_sy, t1, optimize=True)
    rdm2_ovoo += -0.5*np.einsum('efmi,efmj,ak->iajk', l2, t2, t1, optimize=True)
    rdm2_ovoo += 0.5*np.einsum('efmi,efmk,aj->iajk', l2, t2, t1, optimize=True)

    rdm2_vvov += np.einsum('ecmn,eami,bn->abic', l2, t2, t1, optimize=True)
    rdm2_vvov += -np.einsum('ecmn,ebmi,an->abic', l2, t2, t1, optimize=True)

    rdm2_vvov += -0.5*np.einsum('ecmn,abmn,ei->abic', l2, tau2_sy, t1, optimize=True)
    rdm2_vvov += 0.5*np.einsum('cemn,bemn,ai->abic', l2, t2, t1, optimize=True)
    rdm2_vvov += -0.5*np.einsum('cemn,aemn,bi->abic', l2, t2, t1, optimize=True)

    #L2tau2**2
    rdm2_vvoo += 0.5*np.einsum('efmn,efmi,abjn->abij', l2, t2, tau2_asy, optimize=True)
    rdm2_vvoo += -0.5*np.einsum('efmn,efmj,abin->abij', l2, t2, tau2_asy, optimize=True)
    rdm2_vvoo += 0.5*np.einsum('efmn,eamn,bfij->abij', l2, t2, tau2_asy, optimize=True)
    rdm2_vvoo += -0.5*np.einsum('efmn,ebmn,afij->abij', l2, t2, tau2_asy, optimize=True)
    rdm2_vvoo += 0.25*np.einsum('efmn,efij,abmn->abij', l2, tau2_sy, tau2_sy, optimize=True)
    rdm2_vvoo += np.einsum('efmn,eami,bfjn->abij', l2, t2, t2, optimize=True)
    rdm2_vvoo += -np.einsum('efmn,eamj,bfin->abij', l2, t2, t2, optimize=True)
    rdm2_vvoo += -np.einsum('efmn,eami,bn,fj->abij', l2, t2, t1, t1, optimize=True)
    rdm2_vvoo += np.einsum('efmn,ebmi,an,fj->abij', l2, t2, t1, t1, optimize=True)
    rdm2_vvoo += np.einsum('efmn,eamj,bn,fi->abij', l2, t2, t1, t1, optimize=True)
    rdm2_vvoo += -np.einsum('efmn,ebmj,an,fi->abij', l2, t2, t1, t1, optimize=True)


    rdm2_oovo = -np.transpose(rdm2_ooov, (0,1,3,2))
    rdm2_ovvo = -np.transpose(rdm2_ovov, (0,1,3,2))
    rdm2_vvvo = -np.transpose(rdm2_vvov, (0,1,3,2))
    rdm2_oooa = np.concatenate((rdm2_oooo, rdm2_ooov), axis=3)
    rdm2_oova = np.concatenate((rdm2_oovo, rdm2_oovv), axis=3)
    rdm2_ooaa = np.concatenate((rdm2_oooa, rdm2_oova), axis=2)
    rdm2_ovoa = np.concatenate((rdm2_ovoo, rdm2_ovov), axis=3)
    rdm2_ovva = np.concatenate((rdm2_ovvo, rdm2_ovvv), axis=3)
    rdm2_ovaa = np.concatenate((rdm2_ovoa, rdm2_ovva), axis=2)
    rdm2_voaa = -np.transpose(rdm2_ovaa, (1,0,2,3))
    rdm2_vvoa = np.concatenate((rdm2_vvoo, rdm2_vvov), axis=3)
    rdm2_vvva = np.concatenate((rdm2_vvvo, rdm2_vvvv), axis=3)
    rdm2_vvaa = np.concatenate((rdm2_vvoa, rdm2_vvva), axis=2)
    rdm2_oaaa = np.concatenate((rdm2_ooaa, rdm2_ovaa), axis=1)
    rdm2_vaaa = np.concatenate((rdm2_voaa, rdm2_vvaa), axis=1)
    rdm2 = np.concatenate((rdm2_oaaa, rdm2_vaaa), axis=0)

    return rdm2