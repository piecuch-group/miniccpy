import numpy as np
from pyscf import ao2mo

from miniccpy.energy import hf_energy, hf_energy_from_fock

def get_integrals_from_pyscf(meanfield):
    """Obtain the molecular orbital integrals from PySCF and convert them to
    the normal-ordered form."""

    molecule = meanfield.mol
    mo_coeff = meanfield.mo_coeff
    norbitals = mo_coeff.shape[1]

    kinetic_aoints = molecule.intor_symmetric("int1e_kin")
    nuclear_aoints = molecule.intor_symmetric("int1e_nuc")
    e1int = np.einsum("pi,pq,qj->ij", mo_coeff, kinetic_aoints + nuclear_aoints, mo_coeff)
    e2int = np.transpose(
        np.reshape(ao2mo.kernel(molecule, mo_coeff, compact=False), 4 * (norbitals,)),
        (0, 2, 1, 3),
    )

    z, g = spatial_to_spinorb(e1int, e2int)
    g -= np.transpose(g, (0, 1, 3, 2))

    occ = slice(0, molecule.nelectron)
    fock = get_fock(z, g, occ)
    e_hf = hf_energy(z, g, occ)
    e_hf_test = hf_energy_from_fock(fock, g, occ)

    assert( abs(e_hf - e_hf_test) < 1.0e-09 )

    return z, g, fock, e_hf + molecule.energy_nuc(), molecule.energy_nuc()

def get_integrals_from_gamess(fcidump, nelectron, norbitals):
    """Obtain the molecular orbital integrals from GAMESS FCIDUMP file."""

    e1int = np.zeros((norbitals, norbitals))
    e2int = np.zeros((norbitals, norbitals, norbitals, norbitals))

    with open(fcidump) as fp:

        ct2body = 0
        ct1body = 0
        for ct, line in enumerate(fp.readlines()):

            if ct < 4: continue

            L = line.split()

            Cf = float(L[0].replace("D", "E"))
            p = int(L[1])-1
            r = int(L[2])-1
            q = int(L[3])-1
            s = int(L[4])-1

            if q !=-1 and s!= -1: # twobody term

                e2int[p,r,q,s] = Cf
                e2int[r,p,q,s] = Cf
                e2int[p,r,s,q] = Cf
                e2int[r,p,s,q] = Cf
                e2int[q,s,p,r] = Cf
                e2int[q,s,r,p] = Cf
                e2int[s,q,p,r] = Cf
                e2int[s,q,r,p] = Cf

            elif q == -1 and s == -1 and p != -1: # onebody term
                e1int[p,r] = Cf
                e1int[r,p] = Cf

            else: # nuclear repulsion
                nuclear_repulsion = Cf

    # Convert from chemist to physics notation
    e2int = e2int.transpose(0, 2, 1, 3)

    z, g = spatial_to_spinorb(e1int, e2int)
    g -= np.transpose(g, (0, 1, 3, 2))

    occ = slice(0, nelectron)
    fock = get_fock(z, g, occ)
    e_hf = hf_energy(z, g, occ)
    e_hf_test = hf_energy_from_fock(fock, g, occ)

    assert( abs(e_hf - e_hf_test) < 1.0e-09 )

    return z, g, fock, e_hf + nuclear_repulsion, nuclear_repulsion

def get_integrals_from_custom_hamiltonian(h1, h2):
    
    norbitals = h1.shape[0]
    nelectron = h1.shape[0]

    # Diagonalize one-body matrix to get Huckel eigenstates as the single-particle spatial orbital (MO) basis
    mo_energy, mo_coeff = np.linalg.eigh(h1)

    # Perform AO to MO transformation in spatial orbital basis 
    e1int = np.einsum("ip,jq,ij->pq", mo_coeff, mo_coeff, h1, optimize=True)
    e2int = np.einsum("ip,jq,kr,ls,ijkl->pqrs", mo_coeff, mo_coeff, mo_coeff, mo_coeff, h2, optimize=True)
    # Convert from spatial orbitals to spin-orbitals
    z, g = spatial_to_spinorb(e1int, e2int)
    # Antisymmetrize two-body integrals
    g -= np.transpose(g, (0, 1, 3, 2))

    # Get correlated slicing arrays
    o = slice(0, nelectron)
    v = slice(nelectron, 2 * norbitals)

    # build Fock matrix and HF energy
    fock = get_fock(z, g, o)
    e_hf = hf_energy(z, g, o)
    e_hf_test = hf_energy_from_fock(fock, g, o)
    assert(abs(e_hf - e_hf_test) < 1.0e-09)

    return z, g, fock, o, v, e_hf


def spatial_to_spinorb(e1int, e2int):
    """Convert spatial orbital integrals to spinorbital integrals."""

    n = e1int.shape[0]
    z = np.zeros((2*n, 2*n))
    g = np.zeros((2*n, 2*n, 2*n, 2*n))

    for i in range(2*n):
        for j in range(2*n):
            if i % 2 == j % 2:
                i0 = int(np.floor(i/2))
                j0 = int(np.floor(j/2))
                z[i, j] = e1int[i0, j0]
    for i in range(2*n):
        for j in range(2*n):
            for k in range(2*n):
                for l in range(2*n):
                    if i % 2 == k % 2 and j % 2 == l % 2:
                        i0 = int(np.floor(i/2))
                        j0 = int(np.floor(j/2))
                        k0 = int(np.floor(k/2))
                        l0 = int(np.floor(l/2))
                        g[i, j, k, l] = e2int[i0, j0, k0, l0]
    return z, g


def get_fock(z, g, o):
    """Calculate the Fock matrix elements defined by
        < p | f | q > = < p | z | q > + < p i | v | q i >
    using the molecular spinorbital integrals."""

    f = z + np.einsum("piqi->pq", g[:, o, :, o])

    return f

