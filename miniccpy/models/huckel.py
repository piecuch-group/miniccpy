import numpy as np
from miniccpy.constants import eV_to_hartree, ang_to_bohr
from miniccpy.integrals import get_integrals_from_custom_hamiltonian
from miniccpy.printing import print_custom_system_information

def mataga_nishimoto(r, gamma):
    """Computes the two-body on-site and nearest-neighbor repulsion of the PPP
    Hamiltonian parameterized using the Mataga-Nishimoto form.

    gamma_ij = <ij|v|ij> = e**2/(r_ij + a), where a = e**2 / gamma
    gamma = IP - EA of carbon atom (equivalent to Hubbard U parameter)
    r_ij = |r_i - r_j|
    
    [see Paldus and Piecuch, IJQC 42, 135 (1992)]."""

    if gamma != 0:
        gamma_ij = 1.0/(r + 1.0/gamma)
    else:
        gamma_ij = 0.0
    
    return gamma_ij

def ppp_hamiltonian(n, cyclic, alpha=0.0, beta=-2.4, gamma=10.84, r=1.4, hubbard=False):
    """Computes the 1-electron and 2-electron parts of the PPP Hamiltonian
    and returns the resulting spinorbital MO integrals using eigenstates of
    the one-electron Huckel part of the PPP Hamiltonian (i.e., Z) as the 
    single-particle basis.
    
    Input:
    ------
       n : Number of electrons, or equivalently, number of C atoms
       cyclic : True/False to specify Hamiltonian or linear or cyclic polyene
       alpha : Huckel on-site one-electron energy (energy of p_z orbital in C); typically set to 0.
       beta : Huckel resonance integral (hopping parameter); typical value is -2.4 eV.
       gamma : Hubbard on-site electron-electron repulsion (U parameter); typical value is 10.84 eV.
       r : Distance between nearest-neighbor C-C bonds; typical value is 1.4 angstrom.
       hubbard : True/False to specify whether to use Hubbard-type Hamiltonian with only on-site 2-electron interactions
    """

    # Model Hamiltonian parameters
    alpha *= eV_to_hartree
    beta *= eV_to_hartree
    gamma *= eV_to_hartree
    r *= ang_to_bohr
   
    # Form the adjacency matrix specifying the connectivity of the polyene
    adjacency = np.diag(np.ones(n)) + np.diag(np.ones(n - 1), -1) + np.diag(np.ones(n - 1), 1)
    if cyclic:
        adjacency[0, -1] = beta
        adjacency[-1, 0] = beta

    # Compute the one-electron Huckel part
    h1 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: # on-site one-electron energy alpha (usually set to 0)
                h1[i, j] = alpha * adjacency[i, j]
            else: # off-diagonal resonance integral / hopping matrix element
                h1[i, j] = beta * adjacency[i, j]

    # Compute the two-electron part, assuming on-site and nearest-neighbor interactions only
    h2 = np.zeros((n, n, n, n))
    for i in range(n):
        h2[i, i, i, i] = mataga_nishimoto(0, gamma)
        # For Hubbard models, only include on-site twobody interaction
        if hubbard: continue
        # PPP models incorporate nearest-neighbor two-body interaction as well
        for j in range(i + 1, n):
            if adjacency[i, j] == 1.0: # only put element if they are nearest neighbors
                h2[i, j, i, j] = mataga_nishimoto(r, gamma)
                h2[j, i, j, i] = h2[i, j, i, j]

    z, g, fock, o, v, e_hf = get_integrals_from_custom_hamiltonian(h1, h2)

    # Print system information
    print_custom_system_information(z, n, 0, e_hf)
    
    energy_1e = np.einsum("ii->", z[o, o])
    energy_2e = 0.5 * np.einsum("ijij->", g[o, o, o, o])
    print("   1e- energy = ", energy_1e)
    print("   2e- energy = ", energy_2e)
    print("")

    return z, g, fock, o, v, e_hf
