import numpy as np

eV_to_hartree = 0.0367493
ang_to_bohr = 1.88973

def mataga_nishimoto(r, gamma00):
    """Computes the two-body on-site and nearest-neighbor repulsion of the PPP
    model parameterized using the Mataga-Nishimoto form,
    gamma_{u,v} = e**2/(R_{uv} + a}, where a = e**2/gamma00.
    gamma00 is the on-site repulsion term (equivalent to Hubbard U) and is given
    by the difference between the valence state IP and EA.
    R_{uv} is the seapration between adjacent sites, and can be approximated by 1.4 A for C-C bonds.
    [see Paldus and Piecuch, IJQC 42, 135 (1992)]."""
    
    a = 1.0/gamma00
    return 1.0/(r + a)

def linear_huckel_model(n):
    # Huckel parameters
    alpha = 0.0
    beta = -2.4 * eV_to_hartree
    # Mataga-Nishimoto parameters
    gamma00 = 10.84 * eV_to_hartree
    r = 1.4 * ang_to_bohr
    
    h1 = np.diag(alpha*np.ones(n)) + np.diag(beta*np.ones(n - 1), -1) + np.diag(beta*np.ones(n - 1), 1)
    h2 = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            if h1[i, j] != 0.0:
                h2[i, j, i, j] = mataga_nishimoto(r, gamma00)

    fock, g, o, v = get_integrals(h1, h2)
        
    return fock, g, o, v, h1, h2

def cyclic_huckel_model(n):
    # Huckel parameters
    alpha = 0.0
    beta = -2.4 * eV_to_hartree
    # Mataga-Nishimoto parameters
    gamma00 = 10.84 * eV_to_hartree
    r = 1.4 * ang_to_bohr
    
    h1 = np.diag(alpha*np.ones(n)) + np.diag(beta*np.ones(n - 1), -1) + np.diag(beta*np.ones(n - 1), 1)
    h1[0, -1] = beta
    h1[-1, 0] = beta
    h2 = np.zeros((n, n, n, n))
    for i in range(n):
        h2[i, i, i, i] = mataga_nishimoto(0, gamma00)
        for j in range(i + 1, n):
            if h1[i, j] != 0.0:
                h2[i, j, i, j] = mataga_nishimoto(r, gamma00)
                h2[j, i, j, i] = mataga_nishimoto(r, gamma00)

    fock, g, o, v = get_integrals(h1, h2)
        
    return fock, g, o, v, h1, h2

def get_integrals(h1, h2):
    
    from miniccpy.integrals import spatial_to_spinorb, get_fock
    from miniccpy.energy import hf_energy, hf_energy_from_fock
    from miniccpy.printing import print_custom_system_information
    
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

    # Print system information
    print_custom_system_information(fock, nelectron, 0, e_hf)
    
    return fock, g, o, v
