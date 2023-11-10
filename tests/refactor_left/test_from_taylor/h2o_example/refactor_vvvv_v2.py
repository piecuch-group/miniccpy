import numpy as np
from miniccpy.driver import run_scf_gamess, run_cc_calc, get_hbar, run_leftcc_calc


fock, g, e_hf, o, v = run_scf_gamess("h2o.FCIDUMP", 20, 50, 0, rhf=True)

T, E_corr = run_cc_calc(fock, g, o, v, method='rccsd', maxit=80, convergence=1.0e-010)
H1, H2 = get_hbar(T, fock, g, o, v, method="rccsd")
L = run_leftcc_calc(T, fock, H1, H2, o, v, method="left_rccsd", maxit=80)

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
)
# apply symmetrizer (ij)(ab)
LH_refactor += LH_refactor.transpose(1, 0, 3, 2)
print("Error = ", np.linalg.norm(LH_exact.flatten() - LH_refactor.flatten()))
print("Sum of LH = ", sum(abs(LH_refactor.flatten())))
