import numpy as np

def linear_huckel_model(n):
    alpha = 0.0
    beta = -1.0
    
    h1 = np.diag(alpha*np.ones(n)) + np.diag(beta*np.ones(n - 1), -1) + np.diag(beta*np.ones(n - 1), 1)
    h2 = np.zeros((n, n, n, n))
    for i in range(n):
        h2[i, i, i, i] = 2.0 * abs(beta)

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
    e1int = np.einsum("ip,ij,jq->pq", mo_coeff, h1, mo_coeff, optimize=True)
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
