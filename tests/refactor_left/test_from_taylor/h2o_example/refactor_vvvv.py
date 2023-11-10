import numpy as np
from miniccpy.driver import run_scf_gamess, run_cc_calc, get_hbar, run_leftcc_calc


fock, g, e_hf, o, v = run_scf_gamess("h2o.FCIDUMP", 20, 50, 0, rhf=True)

T, E_corr = run_cc_calc(fock, g, o, v, method='rccsd', maxit=80, convergence=1.0e-010)
H1, H2 = get_hbar(T, fock, g, o, v, method="rccsd")
L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_rccsd", maxit=0)

t1, t2 = T
l1, l2 = L

nu, no = t1.shape

tau = t2 + np.einsum("ai,bj->abij", t1, t1)
H2[v, v, v, v] = g[v, v, v, v] + (
        - np.einsum("bmfe,am->abef", g[v, o, v, v], t1, optimize=True)
        - np.einsum("amef,bm->abef", g[v, o, v, v], t1, optimize=True)
        + np.einsum("mnef,abmn->abef", g[o, o, v, v], tau, optimize=True)
)
LH_exact = 0.5 * np.einsum("efab,efij->abij", H2[v, v, v, v], l2, optimize=True)
# apply symmetrizer (ij)(ab)
LH_exact += LH_exact.transpose(1, 0, 3, 2)


# Refactored scheme
LH_refactor = np.zeros((nu, nu, no, no)) 
tau = t2 + np.einsum("ai,bj->abij", t1, t1)    # CPU: no^2 nu^2, Memory: no^2 nu^2
I_oooo = np.einsum("efij,efmn->mnij", l2, tau) # CPU: nu^2 no^4, Memory: no^2 nu^2 
I_vooo = np.einsum("efij,fn->enij", l2, t1)    # CPU: nu^2 no^3, Memory: no^2 nu^2
LH_refactor = (
                  0.5 * np.einsum("efab,efij->abij", g[v, v, v, v], l2, optimize=True)
                + 0.5 * np.einsum("mnab,mnij->abij", g[o, o, v, v], I_oooo, optimize=True)
                - np.einsum("emba,emji->abij", g[v, o, v, v], I_vooo, optimize=True)
                #- np.einsum("emab,emij->abij", g[v, o, v, v], I_vooo, optimize=True)
)
#bare_vvvv = 0.5 * np.einsum("efab,efij->abij", g[v, v, v, v], l2, optimize=True)
#vvvv_oooo_term = 0.5 * np.einsum("mnab,mnij->abij", g[o, o, v, v], I_oooo, optimize=True)
# v(enab) * rl(enij) -> L2(abij)
#for n in range(no):
#    # v(eab) * rl(eij) -> L2(abij)
#    g_vvv = g[v, o, v, v][:, n, :, :]
#    #print("n = ", n + 1, "VE NORM = ", sum(abs(g_vvv.flatten())))
#    #print("n = ", n + 1, "RL(IOF1:IOF2) NORM = ", sum(abs(0.5 * I_vooo[:, n, :, :].flatten())))
#    LH_refactor -= np.einsum("eab,eij->abij", g_vvv, I_vooo[:, n, :, :])
#    vvvv_oooo_term -= np.einsum("eab,eij->abij", g_vvv, I_vooo[:, n, :, :])
#    LH_refactor -= np.einsum("abe,eij->abij", g_vvv, I_vooo[:, n, :, :])
#    #print("n = ", n + 1, "L2 NORM = ", sum(abs(LH_refactor.flatten())))
# apply symmetrizer (ij)(ab)
LH_refactor += LH_refactor.transpose(1, 0, 3, 2)
#bare_vvvv += bare_vvvv.transpose(1, 0, 3, 2)
#vvvv_oooo_term += vvvv_oooo_term.transpose(1, 0, 3, 2)
print("Error = ", np.linalg.norm(LH_exact.flatten() - LH_refactor.flatten()))

print("Sum of LH = ", sum(abs(LH_refactor.flatten())))
#print("Sum of vvvv = ", sum(abs(bare_vvvv.flatten())))
#print("Sum of oooo = ", sum(abs(vvvv_oooo_term.flatten())))




#print("")
#I_vooo = np.einsum("efij,fn->enij", l2, t1)    # CPU: nu^2 no^3, Memory: no^2 nu^2
#arr = 0.5 * I_vooo.transpose(2, 3, 1, 0)
#temp = 0.0
#thresh = 0.0
#for i in range(no):
#    for j in range(no):
#        for n in range(no):
#            for e in range(nu):
#                if abs(arr[i, j, n, e]) > thresh:
#                    print("I_ooov({}, {}, {}, {}) = {}".format(i + 1, j + 1, n + 1, e + 1, arr[i, j, n, e]))
