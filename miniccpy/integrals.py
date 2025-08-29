import numpy as np
from pyscf import ao2mo

from miniccpy.energy import hf_energy, hf_energy_from_fock, rhf_energy

def get_integrals_from_pyscf(meanfield):
    """Obtain the RHF/ROHF molecular orbital integrals from PySCF and convert them to
    the normal-ordered form."""

    molecule = meanfield.mol
    mo_coeff = meanfield.mo_coeff
    norbitals = mo_coeff.shape[1]

    e1int = np.einsum("pi,pq,qj->ij", mo_coeff, meanfield.get_hcore(), mo_coeff)
    e2int = np.transpose(
        np.reshape(ao2mo.kernel(molecule, mo_coeff, compact=False), 4 * (norbitals,)),
        (0, 2, 1, 3),
    )

    z, g = spatial_to_spinorb(e1int, e2int)
    g -= np.transpose(g, (0, 1, 3, 2))

    # reorder integrals to be docc, socc_a, unocc_b, virt
    z, g, _ = reorder_occ_first(meanfield, z, g)

    occ = slice(0, molecule.nelectron)

    fock = get_fock(z, g, occ)
    e_hf = hf_energy(z, g, occ)
    e_hf_test = hf_energy_from_fock(fock, g, occ)

    assert( abs(e_hf - e_hf_test) < 1.0e-09 )

    return z, g, fock, e_hf + molecule.energy_nuc(), molecule.energy_nuc()


def reorder_occ_first(meanfield, z, g):
    """
    Reorder spin–orbital integrals so that occupied spin–orbitals are a
    contiguous block. Works for RHF and ROHF (single set of spatial MOs).
    """
    nspin = z.shape[0]           # = 2 * nmo
    nmo = nspin // 2
    mo_occ = np.asarray(meanfield.mo_occ)  # length nmo, entries in {2,1,0} for ROHF

    # Decide which spin is occupied for singly-occupied spatial MOs.
    na, nb = meanfield.mol.nelec
    singles_spin = 0 if na >= nb else 1      # 0: alpha, 1: beta

    # Build occupied spin–orbital indices in the current (αβ interleaved) order.
    occ_idx = []
    for p, occ in enumerate(mo_occ):
        if occ > 1.5:                         # doubly occupied
            occ_idx.extend([2*p, 2*p+1])      # α and β
        elif occ > 0.5:                       # singly occupied
            occ_idx.append(2*p + singles_spin)

    occ_idx = np.array(occ_idx, dtype=int)
    assert len(occ_idx) == meanfield.mol.nelectron

    # Put all remaining spin–orbitals after the occupied block (preserving order).
    all_idx = np.arange(nspin)
    virt_idx = all_idx[~np.isin(all_idx, occ_idx, assume_unique=False)]
    perm = np.concatenate([occ_idx, virt_idx])

    # Apply the permutation to 1e and 2e integrals.
    z_reo = z[np.ix_(perm, perm)]
    g_reo = g[np.ix_(perm, perm, perm, perm)]
    return z_reo, g_reo, perm


def get_integrals_from_pyscf_rhf(meanfield):
    """Obtain the spatial molecular orbital integrals from PySCF and convert them to
    the normal-ordered form. This implementation is used for RHF-based nonorthogonally
    spin-adapted methods."""

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

    occ = slice(0, int(molecule.nelectron / 2))
    fock = get_fock_rhf(e1int, e2int, occ)
    e_hf = rhf_energy(e1int, e2int, occ)

    return e1int, e2int, fock, e_hf + molecule.energy_nuc(), molecule.energy_nuc()

def get_integrals_from_pyscf_uhf(meanfield):
    """Obtain the UHF molecular orbital integrals from PySCF and convert them to
    the normal-ordered form."""

    molecule = meanfield.mol
    mo_coeff_a, mo_coeff_b = meanfield.mo_coeff

    kinetic_aoints = molecule.intor_symmetric("int1e_kin")
    nuclear_aoints = molecule.intor_symmetric("int1e_nuc")
    hcore_aoints = kinetic_aoints + nuclear_aoints
    eri_aoints = np.transpose(molecule.intor("int2e", aosym="s1"), (0, 2, 1, 3))

    # Transform 1-body integrals
    e1int_a = np.einsum("pi,pq,qj->ij", mo_coeff_a, hcore_aoints, mo_coeff_a)
    e1int_b = np.einsum("pi,pq,qj->ij", mo_coeff_b, hcore_aoints, mo_coeff_b)
    # Transform 2-body integrals
    e2int_aa = np.einsum("pi,qj,rk,sl,pqrs->ijkl", mo_coeff_a, mo_coeff_a, mo_coeff_a, mo_coeff_a, eri_aoints, optimize=True)
    e2int_ab = np.einsum("pi,qj,rk,sl,pqrs->ijkl", mo_coeff_a, mo_coeff_b, mo_coeff_a, mo_coeff_b, eri_aoints, optimize=True)
    e2int_bb = np.einsum("pi,qj,rk,sl,pqrs->ijkl", mo_coeff_b, mo_coeff_b, mo_coeff_b, mo_coeff_b, eri_aoints, optimize=True)

    z, g = spatial_to_spinorb_uhf(e1int_a, e1int_b, e2int_aa, e2int_ab, e2int_bb)
    g -= np.transpose(g, (0, 1, 3, 2))

    occ = slice(0, molecule.nelectron)
    fock = get_fock(z, g, occ)
    e_hf = hf_energy(z, g, occ)
    e_hf_test = hf_energy_from_fock(fock, g, occ)

    assert( abs(e_hf - e_hf_test) < 1.0e-09 )

    return z, g, fock, e_hf + molecule.energy_nuc(), molecule.energy_nuc()

def get_integrals_from_gamess(fcidump, nelectron, norbitals, rhf=False):
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

    if rhf:
        occ = slice(0, int(nelectron / 2))
        z = e1int
        g = e2int
        fock = get_fock_rhf(e1int, e2int, occ)
        e_hf = rhf_energy(e1int, e2int, occ)
    else:
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

def spatial_to_spinorb_onebody(e1int):
    """Convert one-body spatial orbital integrals to spinorbital integrals."""

    n = e1int.shape[0]
    z = np.zeros((2*n, 2*n))

    for i in range(2*n):
        for j in range(2*n):
            if i % 2 == j % 2:
                i0 = int(np.floor(i/2))
                j0 = int(np.floor(j/2))
                z[i, j] = e1int[i0, j0]
    return z

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

def spatial_to_spinorb_uhf(e1int_a, e1int_b, e2int_aa, e2int_ab, e2int_bb):
    """Convert the UHF-transformed spatial integrals to spinorbital integrals."""

    n = e1int_a.shape[0]
    z = np.zeros((2*n, 2*n))
    g = np.zeros((2*n, 2*n, 2*n, 2*n))

    for i in range(2*n):
        for j in range(2*n):
            if i % 2 == 0 and j % 2 == 0:
                z[i, j] = e1int_a[i // 2, j // 2]
            elif i % 2 == 1 and j % 2 == 1:
                z[i, j] = e1int_b[(i - 1) // 2, (j - 1) // 2]

    for i in range(2*n):
        for j in range(2*n):
            for k in range(2*n):
                for l in range(2*n):
                    # aaaa
                    if i % 2 == 0 and j % 2 == 0 and k % 2 == 0 and l % 2 == 0:
                        g[i, j, k, l] = e2int_aa[i // 2, j // 2, k // 2, l // 2]
                    # bbbb
                    elif i % 2 == 1 and j % 2 == 1 and k % 2 == 1 and l % 2 == 1:
                        g[i, j, k, l] = e2int_bb[(i - 1) // 2, (j - 1) // 2, (k - 1) // 2, (l - 1) // 2]
                    # abab
                    elif i % 2 == 0 and j % 2 == 1 and k % 2 == 0 and l % 2 == 1:
                        g[i, j, k, l] = e2int_ab[i // 2, (j - 1) // 2, k // 2, (l - 1) // 2]
                    # baba
                    elif i % 2 == 1 and j % 2 == 0 and k % 2 == 1 and l % 2 == 0:
                        g[i, j, k, l] = e2int_ab[(i - 1) // 2, j // 2, (k - 1) // 2, l // 2]
                    # abba
                    elif i % 2 == 0 and j % 2 == 1 and k % 2 == 1 and l % 2 == 0:
                        g[i, j, k, l] = -e2int_ab[i // 2, (j - 1) // 2, (k - 1) // 2, l // 2]
                    # baab
                    elif i % 2 == 1 and j % 2 == 0 and k % 2 == 0 and l % 2 == 1:
                        g[i, j, k, l] = -e2int_ab[(i - 1) // 2, j // 2, k // 2, (l - 1) // 2]
    return z, g

def get_fock(z, g, o):
    """Calculate the Fock matrix elements defined by
        < p | f | q > = < p | z | q > + < p i | v | q i >
    using the molecular spinorbital integrals."""

    f = z + np.einsum("piqi->pq", g[:, o, :, o])

    return f

def get_fock_rhf(z, g, o):
    """Calculate the RHF Fock matrix elements defined by
        < p | f | q > = < p | z | q > + 2 < p i | v | q i > - < p i | v | i q >
    using the spin-free orbital integrals z and v."""

    f = (
            z + 2.0 * np.einsum("piqi->pq", g[:, o, :, o])
              - np.einsum("piiq->pq", g[:, o, o, :])
    )
    return f
