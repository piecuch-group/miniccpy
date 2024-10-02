import numpy as np
from miniccpy.integrals import spatial_to_spinorb_onebody

def get_multipole_integrals(l, mol, mf):
    print(f"   Computing L = {l} Multipole Integrals using PySCF")
    nao = mol.nao
    # Dipole
    if l == 1:
        Q_ao = mol.intor("int1e_r").reshape(3, nao, nao)
        Q_mo = np.einsum("xij,ip,jq->xpq", Q_ao, mf.mo_coeff, mf.mo_coeff, optimize=True)
        mu = np.zeros((3, 2*nao, 2*nao))
        for i in range(3):
            mu[i, :, :] = spatial_to_spinorb_onebody(Q_mo[i, :, :])
    # Quadrupole
    elif l == 2:
        Q_ao = mol.intor('int1e_rr').reshape(3, 3, nao, nao)
        Q_mo = np.einsum("xyij,ip,jq->xypq", Q_ao, mf.mo_coeff, mf.mo_coeff, optimize=True)
        mu = np.zeros((3, 3, 2*nao, 2*nao))
        for i in range(3):
            for j in range(3):
                mu[i, j, :, :] = spatial_to_spinorb_onebody(Q_mo[i, j, :, :])
    # Octopole
    elif l == 3:
        Q_ao = mol.intor('int1e_rrr').reshape(3, 3, nao, nao)
        Q_mo = np.einsum("xyzij,ip,jq->xyzpq", Q_ao, mf.mo_coeff, mf.mo_coeff, optimize=True)
        mu = np.zeros((3, 3, 3, 2*nao, 2*nao))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    mu[i, j, k, :, :] = spatial_to_spinorb_onebody(Q_mo[i, j, k, :, :])
    # Hexadecapole
    elif l == 4:
        Q_ao = mol.intor('int1e_rrrr').reshape(3, 3, 3, 3, nao, nao)
        Q_mo = np.einsum("xyzwij,ip,jq->xyzwpq", Q_ao, mf.mo_coeff, mf.mo_coeff, optimize=True)
        mu = np.zeros((3, 3, 3, 3, 2*nao, 2*nao))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for u in range(3):
                        mu[i, j, k, u, :, :] = spatial_to_spinorb_onebody(Q_mo[i, j, k, u, :, :])

    else:
        print(f"Angular momentum {l} not supported in PySCF.")

    return mu

