import numpy as np
from miniccpy.driver import run_scf_gamess, run_cc_calc, get_hbar, run_leftcc_calc


fock, g, e_hf, o, v = run_scf_gamess("h2o.FCIDUMP", 20, 50, 0, rhf=True)

T, E_corr = run_cc_calc(fock, g, o, v, method='rccsd', maxit=80, convergence=1.0e-010)
H1, H2 = get_hbar(T, fock, g, o, v, method="rccsd")
L = run_leftcc_calc(T, H1, H2, o, v, method="left_rccsd", maxit=0)

t1, t2 = T
l1, l2 = L

nu, no = t1.shape

Q1 = g[v, o, v, o] - np.einsum("nmei,bn->bmei", g[o, o, v, o], t1, optimize=True)
Q1 = -np.einsum("bmei,am->baei", Q1, t1, optimize=True)
h2_vvvo_exact = g[v, v, v, o] + Q1 + (
            - np.einsum("me,bami->baei", H1[o, v], t2, optimize=True)
            + np.einsum("baef,fi->baei", H2[v, v, v, v], t1, optimize=True)
            + 2.0 * np.einsum("bnef,fani->baei", H2[v, o, v, v], t2, optimize=True)
            - np.einsum("bnfe,fani->baei", H2[v, o, v, v], t2, optimize=True)
            - np.einsum("bnef,fain->baei", H2[v, o, v, v], t2, optimize=True)
            - np.einsum("amfe,bfmi->baei", H2[v, o, v, v], t2, optimize=True)
            - np.einsum("anie,bn->baei", g[v, o, o, v], t1, optimize=True)
            + np.einsum("nmei,banm->baei", g[o, o, v, o], t2, optimize=True)
)

# Refactored scheme
h2_vvvo = np.zeros((nu, nu, nu, no))
Q1 = g[v, o, v, o] - np.einsum("nmei,bn->bmei", g[o, o, v, o], t1, optimize=True)
Q1 = -np.einsum("bmei,am->baei", Q1, t1, optimize=True)
H2[v, v, v, o] = g[v, v, v, o] + Q1 + (
            - np.einsum("me,bami->baei", H1[o, v], t2, optimize=True)
            + np.einsum("baef,fi->baei", H2[v, v, v, v], t1, optimize=True)
            + np.einsum("bnef,fani->baei", H2[v, o, v, v], t2, optimize=True)
            - np.einsum("bnfe,fani->baei", H2[v, o, v, v], t2, optimize=True)
            + np.einsum("bnef,fani->baei", H2[v, o, v, v], t2s, optimize=True)
            - np.einsum("maef,bfmi->baei", H2[o, v, v, v], t2, optimize=True)
            - np.einsum("naei,bn->baei", g[o, v, v, o], t1, optimize=True)
            + np.einsum("nmei,banm->baei", g[o, o, v, o], t2, optimize=True)
)

print("Error = ", np.linalg.norm(h2_vvvo_exact.flatten() - h2_vvvo.flatten()))

