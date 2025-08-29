import numpy as np

def rdm1_ccsd(T, L):

    t1, t2 = T
    l1, l2 = L

    # oo block: gamma_ij --i--.--j--
    rdm1_oo = (
                    -np.einsum('ei,ej->ij', l1, t1, optimize=True)
                    -0.5*np.einsum('efim,efjm->ij', l2, t2, optimize=True)
              )

    #ov block: gamma_ia --ia--.
    rdm1_ov = l1.transpose()

    #vo block: gamma_ai .--ia--
    rdm1_vo = (
                    +np.einsum('em,aeim->ai', l1, t2, optimize=True)
                    -np.einsum('em,ei,am->ai', l1, t1, t1, optimize=True)
                    -0.5*np.einsum('efmn,efin,am->ai', l2, t2, t1, optimize=True)
                    -0.5*np.einsum('efmn,afmn,ei->ai', l2, t2, t1, optimize=True)
                )
    rdm1_vo += t1.copy()

    #vv block: gamma_ab --b--.--a--
    rdm1_vv = (
                    +np.einsum('bm,am->ab', l1, t1, optimize=True)
                    +0.5*np.einsum('bemn,aemn->ab', l2, t2, optimize=True)
                )

    rdm1_oa = np.concatenate((rdm1_oo, rdm1_ov), axis=1)
    rdm1_va = np.concatenate((rdm1_vo, rdm1_vv), axis=1)
    rdm1 = np.concatenate((rdm1_oa, rdm1_va), axis=0)
    return rdm1

def rdm1_ccsdt(istate, jstate, L, T, R=None, r0=None):

    # Ensure that R is passed in for excited-state density matrices
    if jstate == 0:
        assert R is None
        assert r0 is None
    else:
        assert R is not None
        assert r0 is not None

    # unpack vectors
    t1, t2, t3 = T
    l1, l2, l3 = L
    if jstate != 0:
        r1, r2, r3 = R

    nu, no = t1.shape
    
    # oo block: <j|gamma|i> => i->-.->-j 
    rdm1_oo = (
                    -np.einsum('ei,ej->ij', l1, t1, optimize=True)
                    -0.5*np.einsum('efin,efjn->ij', l2, t2, optimize=True)
                    -(1.0/12.0)*np.einsum('efgino,efgjno->ij', l3, t3, optimize=True)
              )
    #vv block: <a|gamma|b> => a-<-.-<-b
    rdm1_vv = (
                    np.einsum('bm,am->ab', l1, t1, optimize=True)
                    +0.5*np.einsum('bfmn,afmn->ab', l2, t2, optimize=True)
                    +(1.0/12.0)*np.einsum('bfgmno,afgmno->ab', l3, t3, optimize=True)
                )
    #vo block: <i|gamma|a> ia>.
    rdm1_ov = l1.transpose()
    #ov block: <a|gamma|i> .<ia
    rdm1_vo = (
                    np.einsum('em,aeim->ai', l1, t2, optimize=True)
                    -np.einsum('em,ei,am->ai', l1, t1, t1, optimize=True)
                    +0.25*np.einsum('efmn,aefimn->ai', l2, t3, optimize=True)
                    -0.5*np.einsum('efmn,afmn,ei->ai', l2, t2, t1, optimize=True)
                    -0.5*np.einsum('efmn,efin,am->ai', l2, t2, t1, optimize=True)
                    -0.25*np.einsum('efgmno,agmo,efin->ai', l3, t2, t2, optimize=True)
                    -(1.0/12.0)*np.einsum('efgmno,afgmno,ei->ai', l3, t3, t1, optimize=True)
                    -(1.0/12.0)*np.einsum('efgmno,efgino,am->ai', l3, t3, t1, optimize=True)
                )

    if istate == jstate:
        rdm1_oo += np.eye(no,dtype=np.float64)
        rdm1_vo += t1.copy()
    if jstate != 0:
        rdm1_oo *= r0
        rdm1_oo += (
                        -np.einsum('ei,ej->ij', l1, r1, optimize=True)
                        -0.5*np.einsum('efin,efjn->ij', l2, r2, optimize=True)
                        -(1.0/12.0)*np.einsum('efgino,efgjno->ij', l3, r3, optimize=True)
                        -np.einsum('efin,ej,fn->ij', l2, t1, r1, optimize=True)
                        -0.25*np.einsum('efgino,ej,fgno->ij', l3, t1, r2, optimize=True)
                        -0.5*np.einsum('efgino,efjn,go->ij', l3, t2, r1, optimize=True)
                    )
        #
        rdm1_vv *= r0
        rdm1_vv += (
                        np.einsum('bm,am->ab', l1, r1, optimize=True)
                        +0.5*np.einsum('bfmn,afmn->ab', l2, r2, optimize=True)
                        +(1.0/12.0)*np.einsum('bfgmno,afgmno->ab', l3, r3, optimize=True)
                        +np.einsum('bfmn,am,fn->ab', l2, t1, r1, optimize=True)
                        +0.25*np.einsum('bfgmno,am,fgno->ab', l3, t1, r2, optimize=True)
                        +0.5*np.einsum('bfgmno,afmn,go->ab', l3, t2, r1, optimize=True)
                    )
        #
        rdm1_ov *= r0
        rdm1_ov += np.einsum('afin,fn->ia', l2, r1, optimize=True) + 0.25*np.einsum('afgino,fgno->ia', l3, r2, optimize=True)
        #
        rdm1_vo *= r0
        rdm1_vo += ( 
                        np.einsum('fn,afin->ai', l1, r2, optimize=True) 
                        +0.25*np.einsum('fgno,afgino->ai', l2, r3, optimize=True) 
                        -np.einsum('em,ei,am->ai', l1, t1, r1, optimize=True) 
                        -np.einsum('em,am,ei->ai', l1, t1, r1, optimize=True) 
                        -0.5*np.einsum('efmn,ei,afmn->ai', l2, t1, r2, optimize=True) 
                        -0.5*np.einsum('efmn,am,efin->ai', l2, t1, r2, optimize=True)
                        -(1.0/12.0)*np.einsum('efgmno,ei,afgmno->ai', l3, t1, r3, optimize=True)
                        -(1.0/12.0)*np.einsum('efgmno,am,efgino->ai', l3, t1, r3, optimize=True)
                        +np.einsum('efmn,aeim,fn->ai', l2, t2, r1, optimize=True)
                        +0.25*np.einsum('efgmno,aeim,fgno->ai', l3, t2, r2, optimize=True)
                        -np.einsum('efmn,ei,am,fn->ai', l2, t1, t1, r1, optimize=True)
                        -0.25*np.einsum('efgmno,ei,am,fgno->ai', l3, t1, t1, r2, optimize=True)
                        -0.5*np.einsum('efmn,efin,am->ai', l2, t2, r1, optimize=True)
                        -0.5*np.einsum('efmn,afmn,ei->ai', l2, t2, r1, optimize=True)
                        -0.25*np.einsum('efgmno,efin,agmo->ai', l3, t2, r2, optimize=True)
                        -0.25*np.einsum('efgmno,afmn,egio->ai', l3, t2, r2, optimize=True)
                        +0.25*np.einsum('efgmno,aefimn,go->ai', l3, t3, r1, optimize=True)
                        -0.5*np.einsum('efgmno,afmn,ei,go->ai', l3, t2, t1, r1, optimize=True)
                        -0.5*np.einsum('efgmno,efin,am,go->ai', l3, t2, t1, r1, optimize=True)
                        -(1.0/12.0)*np.einsum('efgmno,efgino,am->ai', l3, t3, r1, optimize=True)
                        -(1.0/12.0)*np.einsum('efgmno,afgmno,ei->ai', l3, t3, r1, optimize=True)
                    )
        if istate == 0:
            rdm1_vo += r1.copy()
        
    rdm1 = np.vstack((np.hstack((rdm1_oo, rdm1_ov)),
                      np.hstack((rdm1_vo, rdm1_vv))))
    return rdm1
